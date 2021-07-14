# Copyright 2021 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import re
from typing import List, Dict, Set

import yaml
from yaml import Loader


def main():
    import subprocess
    import pathlib
    from collections import Counter, defaultdict

    # run gen-pipeline.sh to get full Buildkite pipeline
    path = pathlib.Path(__file__).parent
    script = path.joinpath('..', '.buildkite', 'gen-pipeline.sh').absolute()
    env = dict(
        PIPELINE_MODE='FULL',
        BUILDKITE_PIPELINE_SLUG='horovod',
        BUILDKITE_PIPELINE_DEFAULT_BRANCH='master',
        BUILDKITE_BRANCH='master'
    )
    proc = subprocess.run([script], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'Script exited with code {proc.returncode}: {script}')

    # parse the pipeline
    pipeline = yaml.load(proc.stdout, Loader=Loader)
    steps = pipeline.get('steps', [])

    images = [plugin['docker-compose#v3.5.0']['build']
              for step in steps if isinstance(step, dict) and 'label' in step
                                and step['label'].startswith(':docker: Build ')
              for plugin in step['plugins'] if 'docker-compose#v3.5.0' in plugin]

    cpu_tests = [(re.sub(r' \(test-.*', '', re.sub(':[^:]*: ', '', step['label'])),
                  step['command'],
                  step['timeout_in_minutes'],
                  plugin['docker-compose#v3.5.0']['run'])
                 for step in steps if isinstance(step, dict) and 'label' in step and 'command' in step
                 and not step['label'].startswith(':docker: Build ') and '-cpu-' in step['label']
                 for plugin in step['plugins'] if 'docker-compose#v3.5.0' in plugin]

    # we need to distinguish the two oneccl variants of some tests
    cpu_tests = [(label + (' [ONECCL OFI]' if 'mpirun_command_ofi' in command else (' [ONECCL MPI]' if 'mpirun_command_mpi' in command else '')),
                  command,
                  timeout,
                  image)
                 for label, command, timeout, image in cpu_tests]

    # check that labels are unique per image
    cardinalities = Counter([(label, image) for label, command, timeout, image in cpu_tests])
    conflicts = [(label, image, card) for (label, image), card in cardinalities.items() if card > 1]
    if conflicts:
        summary = '\n'.join([f'"{label}" for image "{image}"' for label, image, card in conflicts])
        raise RuntimeError(f'There are {len(conflicts)} duplicate test labels for images:\n{summary}')

    # commands for some labels may differ
    # we make their labels unique here
    label_commands = defaultdict(Counter)
    for label, command, timeout, image in cpu_tests:
        label_commands[label][command] += 1

    labels_with_multiple_commands = {label: c for label, c in label_commands.items() if len(c) > 1}
    new_labels_per_label_command = {(label, command): f'{label} {index+1}'
                                    for label, commands in labels_with_multiple_commands.items()
                                    for index, command in enumerate(commands)}

    cpu_tests = [(new_labels_per_label_command[(label, command)] if (label, command) in new_labels_per_label_command else label,
                  command,
                  timeout,
                  image)
                 for label, command, timeout, image in cpu_tests]

    # come up with test ids from test labels
    test_labels = {label for label, command, timeout, image in cpu_tests}
    test_id_per_label = [(label, re.sub('[^a-zA-Z0-9_]', '', re.sub('[ .]', '_', label)))
                         for label in test_labels]
    if len({id for label, id in test_id_per_label}) != len(test_labels):
        raise RuntimeError('Some test ids are not unique')
    test_id_per_label = dict(test_id_per_label)

    # collect tests per image
    tests_per_image = {image: {test_id_per_label[label]
                               for label, command, timeout, test_image in cpu_tests
                               if test_image == image}
                       for image in sorted(images)}
    # define no tests for any image (used for GPU builds below)
    no_tests_per_image = defaultdict(lambda: set())

    # index tests by id
    tests = {test_id_per_label[label]: dict(label=label, command=command, timeout=timeout)
             for label, command, timeout, image in cpu_tests}

    def workflow_header() -> str:
        return (f'# Do not edit this file! It has been generated by .github/gen-workflow-ci.py\n'
                f'\n'
                f'name: CI\n'
                f'\n'
                f'on:\n'
                f'  schedule:\n'
                f'    - cron: \'0 10 * * *\' # everyday at 10am\n'
                f'  push:\n'
                f'    # only consider push to master and tags\n'
                f'    # otherwise modify job.config.outputs.push\n'
                f'    branches: [ master ]\n'
                f'    tags: [ \'v*.*.*\' ]\n'
                f'  pull_request:\n'
                f'    branches: [ master ]\n'
                f'\n'
                f'concurrency:\n'
                f'  # github.ref means something like refs/heads/master or refs/tags/v0.22.1 or the branch.\n'
                f'  # This helps to not cancel concurrent runs on master and a tag that share the same commit\n'
                f'  # On master, head_ref is empty, so we use the SHA of the commit, this means\n'
                f'  # individual commits to master will not be cancelled, but tagged\n'
                f'  group: ci-${{{{ github.ref }}}}-${{{{ github.head_ref || github.sha }}}}\n'
                f'  cancel-in-progress: true\n'
                f'\n')

    def jobs(*jobs: str) -> str:
        return 'jobs:\n' \
               '  debug:\n' \
               '    runs-on: ubuntu-latest\n' \
               '    steps:\n' \
               '    - name: Debug Action\n' \
               '      uses: hmarr/debug-action@v1.0.0\n' + '\n'.join(jobs)

    def validate_workflow_job() -> str:
        return (f'  init-workflow:\n'
                f'    name: "Init Workflow"\n'
                f'    runs-on: ubuntu-latest\n'
                f'    outputs:\n'
                f'      run_builds_and_tests: ${{{{ steps.tests.output.needed }}}}\n'
                f'\n'
                f'    steps:\n'
                f'      - name: Checkout\n'
                f'        uses: actions/checkout@v2\n'
                f'      - name: Setup Python\n'
                f'        uses: actions/setup-python@v2\n'
                f'        with:\n'
                f'          python-version: 3.8\n'
                f'      - name: Pip install dependencies\n'
                f'        run: pip install -r .github/requirements.txt\n'
                f'\n'
                f'      - name: Check ci.yaml is up-to-date\n'
                f'        run: |\n'
                f'          python .github/gen-workflow-ci.py\n'
                f'          if [[ $(git diff .github/workflows/ci.yaml | wc -l) -gt 0 ]]\n'
                f'          then\n'
                f'            echo "::error::Workflow file .github/workflows/ci.yaml is out-dated, please run .github/gen-workflow-ci.py and commit changes"\n'
                f'            exit 1\n'
                f'          fi\n'
                f'        shell: bash\n'
                f'\n'
                f'      - name: Check if tests are needed\n'
                f'        id: tests\n'
                f'        env:\n'
                f'          GITHUB_BASE: ${{{{ github.event.pull_request.base.sha }}}}\n'
                f'          GITHUB_HEAD: ${{{{ github.event.pull_request.head.sha }}}}\n'
                f'        run: |\n'
                f'          if [[ "${{{{ github.event_name }}}}" == "pull_request" ]] && [[ -z "$(python .github/get-changed-code-files.py)" ]]\n'
                f'          then\n'
                f'            echo "::set-output name=needed::false"\n'
                f'            exit 0\n'
                f'          fi\n'
                f'          echo "::set-output name=needed::true"\n')

    def build_and_test_images(id: str,
                              name: str,
                              needs: List[str],
                              images: List[str],
                              parallel_images: str,
                              tests_per_image: Dict[str, Set[str]],
                              tests: Dict[str, Dict],
                              attempts: int = 3) -> str:
        if 'init-workflow' not in needs:
            needs.insert(0, 'init-workflow')
        failure = "'failure'"
        return (f'  {id}:\n'
                f'    name: "{name} (${{{{ matrix.image }}}})"\n'
                f'    needs: [{", ".join(needs)}]\n'
                f'    if: needs.init-workflow.outputs.run_builds_and_tests != \'false\'\n'
                f'    runs-on: ubuntu-latest\n'
                f'\n'
                f'    strategy:\n'
                f'      max-parallel: {len([image for image in images if parallel_images in image])}\n'
                f'      fail-fast: false\n'
                f'      matrix:\n'
                f'        include:\n' +
                '\n'.join([f'          - image: {image}\n' +
                           f''.join([f'            {test}: true\n'
                                     for test in sorted(list(tests_per_image[image]))]) +
                           f'            build_timeout: {30 if "-cpu-" in image else 40}\n'
                           for image in sorted(images)
                           # oneccl does not compile on GitHub Workflows:
                           # https://github.com/horovod/horovod/issues/2846
                           if '-oneccl-' not in image]) +
                f'\n'
                f'    steps:\n'
                f'      - name: Clean up disk space\n'
                f'        # deleting these paths frees 38 GB disk space:\n'
                f'        #   sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc\n'
                f'        # but this sometimes takes 3-4 minutes\n'
                f'        # so we delete only some sub-paths which are known to be quick (10s) and 20 GB\n'
                f'        run: |\n'
                f'          echo ::group::Disk space before clean up\n'
                f'          df -h\n'
                f'          echo ::endgroup::\n'
                f'\n'
                f'          for dir in /usr/share/dotnet/sdk/\*/nuGetPackagesArchive.lzma \\\n'
                f'                     /usr/share/dotnet/shared \\\n'
                f'                     /usr/local/lib/android/sdk/ndk \\\n'
                f'                     /usr/local/lib/android/sdk/build-tools \\\n'
                f'                     /opt/ghc\n'
                f'          do\n'
                f'            echo ::group::Deleting "$dir"\n'
                f'            sudo du -hsc $dir | tail -n1 || true\n'
                f'            sudo rm -rf $dir\n'
                f'            echo ::endgroup::\n'
                f'          done\n'
                f'\n'
                f'          echo ::group::Disk space after clean up\n'
                f'          df -h\n'
                f'          echo ::endgroup::\n'
                f'\n'
                f'      - name: Checkout\n'
                f'        uses: actions/checkout@v2\n'
                f'        with:\n'
                f'          submodules: recursive\n'
                f'\n'
                f'      - name: Setup Python\n'
                f'        uses: actions/setup-python@v2\n'
                f'        with:\n'
                f'          python-version: 3.8\n'
                f'\n'
                f'      - name: Setup docker-compose\n'
                f'        run: pip install docker-compose\n'
                f'\n'
                f'      - name: Configure AWS credentials\n'
                f'        id: aws\n'
                f'        uses: aws-actions/configure-aws-credentials@v1\n'
                f'        # AWS credentials are used to authenticate against AWS ECR to pull and push test images\n'
                f'        # We can only authenticate when running on Horovod repo (not a fork)\n'
                f'        if: github.repository == \'horovod/horovod\'\n'
                f'        continue-on-error: true\n'
                f'        with:\n'
                f'          aws-access-key-id: ${{{{ secrets.AWS_ACCESS_KEY_ID }}}}\n'
                f'          aws-secret-access-key: ${{{{ secrets.AWS_SECRET_ACCESS_KEY }}}}\n'
                f'          aws-region: us-east-1\n'
                f'\n'
                f'      - name: Login to Amazon ECR\n'
                f'        id: ecr\n'
                f'        if: steps.aws.outcome == \'success\'\n'
                f'        continue-on-error: true\n'
                f'        uses: aws-actions/amazon-ecr-login@v1\n'
                f'\n'
                f'      - name: Add cache_from to docker-compose YAML\n'
                f'        if: steps.ecr.outcome == \'success\'\n'
                f'        run: |\n'
                f'          cat > docker-compose.test.override.yml <<EOF\n'
                f'          version: \'2.3\'\n'
                f'          services:\n'
                f'            ${{{{ matrix.image }}}}:\n'
                f'              build:\n'
                f'                cache_from:\n'
                f'                  - ${{{{ steps.ecr.outputs.registry }}}}/buildkite:horovod-${{{{ matrix.image }}}}-latest\n'
                f'          EOF\n'
                f'          cat docker-compose.test.override.yml\n'
                f'        shell: bash\n'
                f'\n'
                f'      - name: Pull latest test image\n'
                f'        if: steps.ecr.outcome == \'success\'\n'
                f'        continue-on-error: true\n'
                f'        run: |\n'
                f'          docker pull ${{{{ steps.ecr.outputs.registry }}}}/buildkite:horovod-${{{{ matrix.image }}}}-latest\n'
                f'        env:\n'
                f'          DOCKER_BUILDKIT: 1\n'
                f'\n'
                f'      - name: Build\n'
                f'        id: build\n'
                f'        run: |\n'
                f'          override_yaml=""\n'
                f'          if [ -e docker-compose.test.override.yml ]; then override_yaml="-f docker-compose.test.override.yml"; fi\n'
                f'          .github/timeout-and-retry.sh ${{{{ matrix.build_timeout }}}}m 3 10 docker-compose -f docker-compose.test.yml $override_yaml build --pull ${{{{ matrix.image }}}}\n'
                f'        env:\n'
                f'          COMPOSE_DOCKER_CLI_BUILD: 1\n'
                f'          DOCKER_BUILDKIT: 1\n'
                f'\n' +
                '\n'.join([f'      - name: "{test["label"]} [attempt {attempt} of {attempts}]"\n'
                           f'        id: {test_id}_{attempt}\n'
                           f'        continue-on-error: {"true" if attempt < attempts else "false"}\n'
                           f'        if: always() && steps.build.outcome == \'success\' && matrix.{test_id} && {"true" if attempt == 1 else f"steps.{test_id}_{attempt-1}.outcome == {failure}"}\n'
                           f'        run: |\n'
                           f'          mkdir -p artifacts/${{{{ matrix.image }}}}/{test_id}_{attempt}\n'
                           f'          docker-compose -f docker-compose.test.yml run -e GITHUB_ACTIONS --rm --volume "$(pwd)/artifacts/${{{{ matrix.image }}}}/{test_id}_{attempt}:/artifacts" ${{{{ matrix.image }}}} /usr/bin/timeout {test["timeout"]}m {test["command"]}\n'
                           f'        shell: bash\n'
                           for test_id, test in sorted(tests.items(), key=lambda test: test[0])
                           for attempt in range(1, attempts+1)]) +
                f'\n'
                f'      - name: Upload Test Results\n'
                f'        uses: actions/upload-artifact@v2\n'
                f'        if: always() && contains(matrix.image, \'-cpu-\')\n'
                f'        with:\n'
                f'          name: Unit Test Results - ${{{{ matrix.image }}}}\n'
                f'          path: artifacts/${{{{ matrix.image }}}}/**/*.xml\n'
                f'\n'
                f'      - name: Push test image\n'
                f'        # We push test image to AWS ECR on push to Horovod master (not a fork)\n'
                f'        if: >\n'
                f'          github.event_name == \'push\' &&\n'
                f'          github.ref == \'refs/heads/master\' &&\n'
                f'          github.repository == \'horovod/horovod\' &&\n'
                f'          steps.ecr.outcome == \'success\'\n'
                f'        continue-on-error: true\n'
                f'        run: |\n'
                f'          docker image ls | head\n'
                f'          docker tag horovod_${{{{ matrix.image }}}} ${{{{ steps.ecr.outputs.registry }}}}/buildkite:horovod-${{{{ matrix.image }}}}-latest\n'
                f'          docker push ${{{{ steps.ecr.outputs.registry }}}}/buildkite:horovod-${{{{ matrix.image }}}}-latest\n'
                f'          docker image ls | head\n'
                f'        shell: bash\n')

    def build_and_test_macos(id: str, name: str, needs: List[str]) -> str:
        if 'init-workflow' not in needs:
            needs.insert(0, 'init-workflow')
        return (f'  {id}:\n'
                f'    name: "{name} (${{{{ matrix.image }}}}-macos)"\n'
                f'    needs: [{", ".join(needs)}]\n'
                f'    if: needs.init-workflow.outputs.run_builds_and_tests != \'false\'\n'
                f'    runs-on: macos-latest\n'
                f'\n'
                f'    strategy:\n'
                f'      max-parallel: 3\n'
                f'      fail-fast: false\n'
                f'      matrix:\n'
                f'        include:\n'
                f''
                f'          - image: test-cpu-openmpi-py3_7-tf1_15_5-keras2_2_4-torch1_2_0-mxnet1_5_0\n'
                f'            HOROVOD_WITH_MPI: 1\n'
                f'            HOROVOD_WITHOUT_GLOO: 1\n'
                f'            TENSORFLOW: 1.15.0\n'
                f'            KERAS: 2.2.4\n'
                f'            PYTORCH: 1.2.0\n'
                f'            PYTORCH_LIGHTNING: 0.7.6\n'
                f'            TORCHVISION: 0.4.0\n'
                f'            MXNET: 1.5.0\n'
                f'\n'
                f'          - image: test-cpu-gloo-py3_8-tf2_2_0-keras2_3_1-torch1_5_0-mxnet1_5_0\n'
                f'            HOROVOD_WITHOUT_MPI: 1\n'
                f'            HOROVOD_WITH_GLOO: 1\n'
                f'            TENSORFLOW: 2.2.0\n'
                f'            KERAS: 2.3.1\n'
                f'            PYTORCH: 1.5.0\n'
                f'            PYTORCH_LIGHTNING: 1.2.9\n'
                f'            TORCHVISION: 0.6.0\n'
                f'            MXNET: 1.5.0\n'
                f'\n'
                f'          - image: test-openmpi-cpu-gloo-py3_8-tf2_3_0-keras2_3_1-torch1_6_0-mxnet1_5_0\n'
                f'            HOROVOD_WITH_MPI: 1\n'
                f'            HOROVOD_WITH_GLOO: 1\n'
                f'            TENSORFLOW: 2.3.0\n'
                f'            KERAS: 2.3.1\n'
                f'            PYTORCH: 1.6.0\n'
                f'            PYTORCH_LIGHTNING: 1.2.9\n'
                f'            TORCHVISION: 0.7.0\n'
                f'            MXNET: 1.5.0\n'
                f'\n'
                f'    steps:\n'
                f'      - name: Checkout\n'
                f'        uses: actions/checkout@v2\n'
                f'        with:\n'
                f'          submodules: recursive\n'
                f'\n'
                f'      - name: Build and Test\n'
                f'        id: build-and-test\n'
                f'        env:\n'
                f'          HOROVOD_WITH_MPI: ${{{{ matrix.HOROVOD_WITH_MPI }}}}\n'
                f'          HOROVOD_WITHOUT_MPI: ${{{{ matrix.HOROVOD_WITHOUT_MPI }}}}\n'
                f'          HOROVOD_WITH_GLOO: ${{{{ matrix.HOROVOD_WITH_GLOO }}}}\n'
                f'          HOROVOD_WITHOUT_GLOO: ${{{{ matrix.HOROVOD_WITHOUT_GLOO }}}}\n'
                f'          TENSORFLOW: ${{{{ matrix.TENSORFLOW }}}}\n'
                f'          KERAS: ${{{{ matrix.KERAS }}}}\n'
                f'          PYTORCH: ${{{{ matrix.PYTORCH }}}}\n'
                f'          PYTORCH_LIGHTNING: ${{{{ matrix.PYTORCH_LIGHTNING }}}}\n'
                f'          TORCHVISION: ${{{{ matrix.TORCHVISION }}}}\n'
                f'          MXNET: ${{{{ matrix.MXNET }}}}\n'
                f'\n'
                f'        run: |\n'
                f'          brew install openmpi cmake libuv pyenv\n'
                f'          export PATH=$(pyenv root)/shims:$PATH\n'
                f'          pyenv install 3.7.7\n'
                f'          pyenv global 3.7.7\n'
                f'          python --version\n'
                f'\n'
                f'          python -m pip install -U pip\n'
                f'          pip install tensorflow==${{TENSORFLOW}} keras==${{KERAS}}\n'
                f'          pip install torch==${{PYTORCH}} pytorch_lightning==${{PYTORCH_LIGHTNING}} torchvision==${{TORCHVISION}}\n'
                f'          pip install mxnet==${{MXNET}}\n'
                f'          HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 pip install --no-cache-dir .[test]\n'
                f'          horovodrun --check-build\n'
                f'\n'
                f'          artifacts_path="$(pwd)/artifacts/${{{{ matrix.image }}}}-macos"\n'
                f'          mkdir -p "$artifacts_path"\n'
                f'          echo "::set-output name=artifacts-path::$artifacts_path"\n'
                f'          echo pytest -v --capture=no --continue-on-collection-errors --junit-xml=$artifacts_path/junit.\$1.\${{HOROVOD_RANK:-\${{OMPI_COMM_WORLD_RANK:-\${{PMI_RANK}}}}}}.\$2.xml \${{@:2}} > pytest.sh\n'
                f'          chmod u+x pytest.sh\n'
                f'\n'
                f'          cd test/parallel\n'
                f'          ls test_*.py | xargs -n 1 horovodrun -np 2 /bin/bash ../../pytest.sh macos\n'
                f'\n'
                f'      - name: Upload Test Results\n'
                f'        uses: actions/upload-artifact@v2\n'
                f'        if: always()\n'
                f'        with:\n'
                f'          name: Unit Test Results - ${{{{ matrix.image }}}}-macos\n'
                f'          path: ${{{{ steps.build-and-test.outputs.artifacts-path }}}}/**/*.xml\n')

    def trigger_buildkite_job(id: str, needs: List[str]) -> str:
        if 'init-workflow' not in needs:
            needs.insert(0, 'init-workflow')
        return (f'  {id}:\n'
                f'    name: "Build and Test GPU (on Builtkite)"\n'
                f'    needs: [{", ".join(needs)}]\n'
                f'    runs-on: ubuntu-latest\n'
                f'    if: >\n'
                f'      needs.init-workflow.outputs.run_builds_and_tests != \'false\' &&\n'
                f'      ( github.event_name == \'push\' || github.event.pull_request.head.repo.full_name == github.repository )\n'
                f'\n'
                f'    steps:\n'
                f'      - name: Trigger Buildkite Pipeline\n'
                f'        id: build\n'
                f'        uses: EnricoMi/trigger-pipeline-action@master\n'
                f'        env:\n'
                f'          PIPELINE: "horovod/horovod"\n'
                # on "push" event, github.event.pull_request.head.ref will be empty
                # and trigger-pipeline-action falls back to github.ref
                f'          BRANCH: "${{{{ github.event.pull_request.head.ref }}}}"\n'
                f'          MESSAGE: "GPU Tests triggered by GitHub"\n'
                f'          BUILDKITE_API_ACCESS_TOKEN: ${{{{ secrets.BUILDKITE_TOKEN }}}}\n'
                f'          BUILD_ENV_VARS: "{{\\"PIPELINE_MODE\\": \\"GPU FULL\\"}}"\n'
                f'\n'
                f'      - name: Download Buildkite Artifacts\n'
                f'        id: download\n'
                f'        uses: docker://ghcr.io/enricomi/download-buildkite-artifact-action:v1\n'
                f'        with:\n'
                f'          github_token: ${{{{ github.token }}}}\n'
                f'          buildkite_token: ${{{{ secrets.BUILDKITE_TOKEN }}}}\n'
                f'          buildkite_build_url: ${{{{ steps.build.outputs.url }}}}\n'
                f'          ignore_build_states: blocked,canceled,skipped,not_run\n'
                f'          ignore_job_states: timed_out\n'
                f'          output_path: artifacts/Unit Test Results - GPUs on Buildkite\n'
                f'\n'
                f'      - name: Upload Test Results\n'
                f'        uses: actions/upload-artifact@v2\n'
                f'        if: always()\n'
                f'        with:\n'
                f'          name: Unit Test Results - GPUs on Builtkite\n'
                f'          path: artifacts/Unit Test Results - GPUs on Buildkite/**/*.xml\n'
                f'\n'
                f'      - name: Escalate Buildkite job state\n'
                f'        if: >\n'
                f'          always() &&\n'
                f'          steps.download.conclusion == \'success\' &&\n'
                f'          steps.download.outputs.build-state != \'passed\'\n'
                f'        run: exit 1\n')

    def publish_unit_test_results(id: str, needs: List[str]) -> str:
        return (f'  {id}:\n'
                f'    name: "Publish Unit Tests Results"\n'
                f'    needs: [{", ".join(needs)}]\n'
                f'    runs-on: ubuntu-latest\n'
                f'    # only run this job when the workflow is in success or failure state,\n'
                f'    # not when it is in cancelled or skipped state\n'
                f'    # only run this job on push events or when the event does not run in a fork repository\n'
                f'    if: >\n'
                f'      ( success() || failure() ) &&\n'
                f'      ( github.event_name == \'push\' || ! github.event.head.repo.fork )\n'
                f'\n'
                f'    steps:\n'
                f'      - name: Download GitHub Artifacts\n'
                f'        uses: actions/download-artifact@v2\n'
                f'        with:\n'
                f'          path: artifacts\n'
                f'\n'
                f'      - name: Publish Unit Test Results\n'
                f'        uses: docker://ghcr.io/enricomi/publish-unit-test-result-action:v1\n'
                f'        if: always()\n'
                f'        with:\n'
                f'          github_token: ${{{{ github.token }}}}\n'
                f'          files: "artifacts/Unit Test Results */**/*.xml"\n')

    def publish_docker_images(needs: List[str], images: List[str]) -> str:
        if 'init-workflow' not in needs:
            needs.insert(0, 'init-workflow')
        if needs != ['init-workflow', 'build-and-test', 'buildkite']:
            raise RuntimeError('This job has hard-coded needs, which you may want to adjust')
        return (f'  docker-config:\n'
                f'    name: Configure docker build\n'
                f'    needs: [{", ".join(needs)}]\n'
                f"    # build-and-test-cpu, build-gpu and buildkite might have been skipped (needs.init-workflow.outputs.run_builds_and_tests == 'false')\n"
                f'    # buildkite might have been skipped (workflow runs for a fork PR)\n'
                f'    # we still want to build docker images in these cases\n'
                f'    if: >\n'
                f'      always() &&\n'
                f"      needs.init-workflow.result == 'success' && (\n"
                f"        needs.init-workflow.outputs.run_builds_and_tests == 'false' ||\n"
                f"        needs.build-and-test.result == 'success' &&\n"
                f"        ( needs.buildkite.result == 'success' || needs.buildkite.result == 'skipped' )\n"
                f'      )\n'
                f'    runs-on: ubuntu-latest\n'
                f'    outputs:\n'
                f'      run: ${{{{ steps.config.outputs.run }}}}\n'
                f'      push: ${{{{ steps.config.outputs.push }}}}\n'
                f'\n'
                f'    steps:\n'
                f'      - name: Config\n'
                f'        id: config\n'
                f'        env:\n'
                f'          # run workflow for all events on Horovod repo and non-schedule events on forks\n'
                f'          run: ${{{{ github.repository == \'horovod/horovod\' || github.event_name != \'schedule\' }}}}\n'
                f'          # push images only from Horovod repo and for schedule and push events\n'
                f'          push: ${{{{ github.repository == \'horovod/horovod\' && contains(\'schedule,push\', github.event_name) }}}}\n'
                f'        run: |\n'
                f'          echo Repository: ${{{{ github.repository }}}}\n'
                f'          echo Event: ${{{{ github.event_name }}}}\n'
                f'          echo Run: $run\n'
                f'          echo "::set-output name=run::$run"\n'
                f'          echo Push: $push\n'
                f'          echo "::set-output name=push::$push"\n'
                f'\n'
                f'  docker-build:\n'
                f'    name: Build docker image ${{{{ matrix.docker-image }}}} (push=${{{{ needs.docker-config.outputs.push }}}})\n'
                f'    needs: docker-config\n'
                f'    if: always() && needs.docker-config.outputs.run == \'true\'\n'
                f'    runs-on: ubuntu-latest\n'
                f'\n'
                f'    # we want an ongoing run of this workflow to be canceled by a later commit\n'
                f'    # so that there is only one concurrent run of this workflow for each branch\n'
                f'    concurrency:\n'
                f'      # github.ref means something like refs/heads/master or refs/tags/v0.22.1 or the branch.\n'
                f'      # This helps to not cancel concurrent runs on master and a tag that share the same commit\n'
                f'      # head_ref refers to the pull request branch so we run only one workflow for the given pull request.\n'
                f'      # On master, head_ref is empty, so we use the SHA of the commit, this means\n'
                f'      # commits to master will not be cancelled, which is important to ensure\n'
                f'      # that every commit to master is full tested and deployed.\n'
                f'      group: docker-${{{{ matrix.docker-image }}}}-${{{{ github.ref }}}}-${{{{ github.head_ref || github.sha }}}}\n'
                f'      cancel-in-progress: true\n'
                f'\n'
                f'    strategy:\n'
                f'      fail-fast: false\n'
                f'      matrix:\n'
                f'        docker-image:\n' +
                ''.join([f'          - {image}\n'
                         for image in images]) +
                f'\n'
                f'    steps:\n'
                f'      - name: Checkout\n'
                f'        uses: actions/checkout@v2\n'
                f'        with:\n'
                f'          submodules: \'recursive\'\n'
                f'\n'
                f'      - name: Docker meta\n'
                f'        id: meta\n'
                f'        uses: crazy-max/ghaction-docker-meta@v2\n'
                f'        with:\n'
                f'          # list of Docker images to use as base name for tags\n'
                f'          images: |\n'
                f'            horovod/${{{{ matrix.docker-image }}}}\n'
                f'          # generate Docker tags based on the following events/attributes\n'
                f'          tags: |\n'
                f'            type=schedule\n'
                f'            type=ref,event=branch\n'
                f'            type=ref,event=pr\n'
                f'            type=semver,pattern={{{{version}}}}\n'
                f'            type=semver,pattern={{{{major}}}}.{{{{minor}}}}\n'
                f'            type=semver,pattern={{{{major}}}}\n'
                f'            type=sha\n'
                f'\n'
                f'      - name: Set up QEMU\n'
                f'        uses: docker/setup-qemu-action@v1\n'
                f'      - name: Set up Docker Buildx\n'
                f'        uses: docker/setup-buildx-action@v1\n'
                f'\n'
                f'      - name: Login to DockerHub\n'
                f'        if: needs.docker-config.outputs.push == \'true\'\n'
                f'        uses: docker/login-action@v1\n'
                f'        with:\n'
                f'          username: ${{{{ secrets.DOCKERHUB_USERNAME }}}}\n'
                f'          password: ${{{{ secrets.DOCKERHUB_TOKEN }}}}\n'
                f'\n'
                f'      - name: Clean up disk space\n'
                f'        # deleting these paths frees 38 GB disk space:\n'
                f'        #   sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc\n'
                f'        # but this sometimes takes 3-4 minutes\n'
                f'        # so we delete only some sub-paths which are known to be quick (10s) and 20 GB\n'
                f'        run: |\n'
                f'          echo ::group::Disk space before clean up\n'
                f'          df -h\n'
                f'          echo ::endgroup::\n'
                f'\n'
                f'          for dir in /usr/share/dotnet/sdk/\*/nuGetPackagesArchive.lzma \\\n'
                f'                     /usr/share/dotnet/shared \\\n'
                f'                     /usr/local/lib/android/sdk/ndk \\\n'
                f'                     /usr/local/lib/android/sdk/build-tools \\\n'
                f'                     /opt/ghc\n'
                f'          do\n'
                f'            echo ::group::Deleting "$dir"\n'
                f'            sudo du -hsc $dir | tail -n1 || true\n'
                f'            sudo rm -rf $dir\n'
                f'            echo ::endgroup::\n'
                f'          done\n'
                f'\n'
                f'          echo ::group::Disk space after clean up\n'
                f'          df -h\n'
                f'          echo ::endgroup::\n'
                f'\n'
                f'      - name: Build and push\n'
                f'        uses: docker/build-push-action@v2\n'
                f'        with:\n'
                f'          context: .\n'
                f'          file: ./docker/${{{{ matrix.docker-image }}}}/Dockerfile\n'
                f'          push: ${{{{ needs.docker-config.outputs.push }}}}\n'
                f'          tags: ${{{{ steps.meta.outputs.tags }}}}\n'
                f'          labels: ${{{{ steps.meta.outputs.labels }}}}\n'
                f'\n'
                f'      - name: Show free space\n'
                f'        if: always()\n'
                f'        run: |\n'
                f'          echo ::group::Disk Space\n'
                f'          df -h\n'
                f'          echo ::endgroup::\n'
                f'\n'
                f'          echo ::group::Docker Space\n'
                f'          docker system df\n'
                f'          echo ::endgroup::\n'
                f'\n'
                f'          echo ::group::Docker Images\n'
                f'          docker images -a\n'
                f'          echo ::endgroup::\n'
                f'\n'
                f'          echo ::group::Docker Container\n'
                f'          docker container list -a\n'
                f'          echo ::endgroup::\n')

    def sync_files(needs: List[str]) -> str:
        return (f'  sync-files:\n'
                f'    name: "Sync Files (${{{{ matrix.name }}}})"\n'
                f'    needs: [{", ".join(needs)}]\n'
                f'    runs-on: ubuntu-latest\n'
                f'\n'
                f'    strategy:\n'
                f'      fail-fast: false\n'
                f'      matrix:\n'
                f'        include:\n'
                f'          - name: Docs Summary\n'
                f'            left_file: README.rst\n'
                f'            right_file: docs/summary.rst\n'
                f'            init: sed -i -e s/docs\///g README.rst\n'
                f'\n'
                f'          - name: Examples Keras Spark3\n'
                f'            left_file: examples/spark/keras/keras_spark_rossmann_run.py\n'
                f'            right_file: examples/spark/keras/keras_spark3_rossmann.py\n'
                f'\n'
                f'    steps:\n'
                f'      - name: Checkout\n'
                f'        uses: actions/checkout@v1\n'
                f'\n'
                f'      - name: Diffing ${{{{ matrix.left_file }}}} with ${{{{ matrix.right_file }}}}\n'
                f'        env:\n'
                f'          LEFT: ${{{{ matrix.left_file }}}}\n'
                f'          RIGHT: ${{{{ matrix.right_file }}}}\n'
                f'          INIT: ${{{{ matrix.init }}}}\n'
                f'        run: |\n'
                f'          $INIT\n'
                f'\n'
                f'          patch --quiet -p0 $LEFT ${{RIGHT}}.patch -o ${{LEFT}}.expected\n'
                f'          if ! diff -q ${{LEFT}}.expected --label $LEFT $RIGHT\n'
                f'          then\n'
                f'            echo\n'
                f'            echo "::error::Files are out-of-sync: $LEFT vs. $RIGHT"\n'
                f'            echo "Unexpected differences are:"\n'
                f'            diff ${{LEFT}}.expected --label $LEFT $RIGHT || true\n'
                f'\n'
                f'            echo\n'
                f'            echo "Use the following as ${{RIGHT}}.patch to accept those changes:"\n'
                f'            diff $LEFT $RIGHT || true\n'
                f'\n'
                f'            false\n'
                f'          fi\n')

    with open(path.joinpath('workflows', 'ci.yaml').absolute(), 'wt') as w:
        heads = ['tfhead', 'torchhead', 'mxnethead']
        release_images = [image for image in images if not all(head in image for head in heads)]
        cpu_release_images = [image for image in release_images if '-cpu-' in image]
        gpu_release_images = [image for image in release_images if '-gpu-' in image or '-mixed-' in image]
        allhead_images = [image for image in images if all(head in image for head in heads)]
        workflow = workflow_header() + jobs(
            validate_workflow_job(),
            # changing these names require changes in the workflow-conclusion step in ci-fork.yaml
            build_and_test_images(id='build-and-test', name='Build and Test', needs=['init-workflow'], images=release_images, parallel_images='-cpu-', tests_per_image=tests_per_image, tests=tests),
            build_and_test_images(id='build-and-test-heads', name='Build and Test heads', needs=['build-and-test'], images=allhead_images, parallel_images='', tests_per_image=tests_per_image, tests=tests),
            build_and_test_macos(id='build-and-test-macos', name='Build and Test macOS', needs=['build-and-test']),
            trigger_buildkite_job(id='buildkite', needs=['build-and-test']),
            publish_unit_test_results(id='publish-test-results', needs=['build-and-test', 'build-and-test-heads', 'build-and-test-macos', 'buildkite']),
            publish_docker_images(needs=['build-and-test', 'buildkite'], images=['horovod', 'horovod-cpu', 'horovod-ray']),
            sync_files(needs=['init-workflow'])
        )
        print(workflow, file=w, end='')


if __name__ == "__main__":
    main()
