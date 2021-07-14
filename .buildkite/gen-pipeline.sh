#!/bin/bash

# exit immediately on failure, or if an undefined variable is used
set -eu

# our repository in AWS
repository=823773083436.dkr.ecr.us-east-1.amazonaws.com/buildkite

# our baseline test is
baseline="test-cpu-gloo-py3_8-tf2_5_0-keras_none-torch1_9_0-mxnet1_8_0_p0-pyspark3_1_2"
# in run_gloo_integration we run 'Elastic Spark * Tests' for this baseline
# so it has to have Gloo mpi kind

# skip tests when there are no code changes
dir="$(dirname "$0")"
code_files=$(python "$dir/get_changed_code_files.py" || echo failure)
tests=$(if [[ "${PIPELINE_MODE:-}" == *"FULL"* ]] && ( [[ "${BUILDKITE_BRANCH:-}" == "${BUILDKITE_PIPELINE_DEFAULT_BRANCH:-}" ]] || [[ -n "$code_files" ]] ); then
  # we vary the baseline along the Python dimension and PySpark together
  # run_gloo_integration expects these to have Gloo mpi kind to run 'Elastic Spark * Tests'
  printf "test-cpu-gloo-py3_7-tf2_5_0-keras_none-torch1_9_0-mxnet1_8_0_p0-pyspark2_4_8 "
  printf "test-cpu-gloo-py3_8-tf2_5_0-keras_none-torch1_9_0-mxnet1_8_0_p0-pyspark3_0_3 "
  # our baseline
  printf "$baseline "

  # then we vary the baseline along mpi kinds dimension
  # our baseline again
# printf "test-cpu-gloo-py3_8-tf2_5_0-keras_none-torch1_9_0-mxnet1_8_0_p0-pyspark3_1_2 "
  printf "test-cpu-mpich-py3_8-tf2_5_0-keras_none-torch1_9_0-mxnet1_8_0_p0-pyspark3_1_2 "
  printf "test-cpu-oneccl-py3_8-tf2_5_0-keras_none-torch1_9_0-mxnet1_8_0_p0-pyspark3_1_2 "
  printf "test-cpu-openmpi-py3_8-tf2_5_0-keras_none-torch1_9_0-mxnet1_8_0_p0-pyspark3_1_2 "
  # note: we test openmpi-gloo mpi kind in this variation in each of [cpu, gpu, mixed]
  printf "test-cpu-openmpi-gloo-py3_8-tf2_5_0-keras_none-torch1_9_0-mxnet1_8_0_p0-pyspark3_1_2 "

  # then we vary the baseline along the framework dimensions all together
  # some frameworks are not available for our baseline Python version 3.8, so we use Python 3.7
  # run_gloo_integration expects tf1 to have Gloo mpi kind to run 'Elastic Spark * Tests'
  printf "test-cpu-gloo-py3_7-tf1_15_5-keras2_2_4-torch1_6_0-mxnet1_5_1_p0-pyspark3_1_2 "
  # there is no mxnet-1.6.0.post0 and mxnet-1.6.0 does not work with horovod
  # https://github.com/apache/incubator-mxnet/issues/16193
  # however, there is an mxnet-cu101-1.6.0.post0, so we test this with gpu instead of cpu
  #printf "test-cpu-gloo-py3_8-tf2_3_2-keras2_3_1-torch1_7_1-mxnet1_6_0_p0-pyspark3_1_2 "
  printf "test-cpu-gloo-py3_8-tf2_4_1-keras2_4_3-torch1_8_1-mxnet1_7_0_p2-pyspark3_1_2 "
  # our baseline again
# printf "test-cpu-gloo-py3_8-tf2_5_0-keras_none-torch1_9_0-mxnet1_8_0_p0-pyspark3_1_2 "
  printf "test-cpu-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_1_2 "

  # then we vary the frameworks for gpu
  printf "test-gpu-gloo-py3_7-tf1_15_5-keras2_2_4-torch1_3_1-mxnet1_5_1_p0-pyspark3_1_2 "
  # this is required as we cannot test mxnet-1.6.0.post0 with cpu
  printf "test-gpu-gloo-py3_8-tf2_3_2-keras2_3_1-torch1_7_1-mxnet1_6_0_p0-pyspark3_1_2 "
  # we additionally test the previous framework combination (CUDA 10.x) with mxnet 1.7.x
  # as mxnet 1.7.x only supports CUDA 10.x, but next framework combination targets CUAA 11.x
  printf "test-gpu-gloo-py3_8-tf2_3_2-keras2_3_1-torch1_7_1-mxnet1_7_0_p1-pyspark3_1_2 "
  # we deviate from mxnet1_7_0_p2 here as other frameworks target CUDA 11.x and
  # mxnet 1.7.x only supports CUDA 10.x, with mxnet 1.8.x we have CUDA 11.x packages
  printf "test-gpu-gloo-py3_8-tf2_4_1-keras2_4_3-torch1_8_1-mxnet1_8_0_p0-pyspark3_1_2 "
  printf "test-gpu-openmpi-gloo-py3_8-tf2_5_0-keras_none-torch1_9_0-mxnet1_8_0_p0-pyspark3_1_2 "
  printf "test-gpu-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_1_2 "

  # and one final test with mixed cpu+gpu
  printf "test-mixed-openmpi-gloo-py3_8-tf2_5_0-keras_none-torch1_9_0-mxnet1_8_0_p0-pyspark3_1_2 "
fi | if [[ "${PIPELINE_MODE:-}" == "GPU FULL" ]]; then sed -E "s/[^ ]*-cpu-[^ ]*//g"; else cat; fi)
read -r -a tests <<< "$tests"


build_test() {
  local test=$1

  echo "- label: ':docker: Build ${test}'"
  echo "  plugins:"
  echo "  - docker-compose#v3.5.0:"
  echo "      build: ${test}"
  echo "      image-repository: ${repository}"
  echo "      cache-from: ${test}:${repository}:${BUILDKITE_PIPELINE_SLUG}-${test}-latest"
  echo "      config: docker-compose.test.yml"
  echo "      push-retries: 5"
  echo "  - ecr#v1.2.0:"
  echo "      login: true"
  echo "  timeout_in_minutes: 30"
  echo "  retry:"
  echo "    automatic: true"
  echo "  agents:"
  echo "    queue: cpu"
}

run_test() {
  local test=$1
  local queue=$2
  local label=$3
  local command=$4
  local timeout=${5-10}

  echo "- label: '${label}'"
  echo "  command: ${command}"
  echo "  artifact_paths: \"artifacts/**\""
  echo "  plugins:"
  echo "  - docker-compose#v3.5.0:"
  echo "      run: ${test}"
  echo "      volumes: \"./artifacts:/artifacts\""
  echo "      config: docker-compose.test.yml"
  echo "      pull-retries: 3"
  echo "  - ecr#v1.2.0:"
  echo "      login: true"
  echo "  timeout_in_minutes: ${timeout}"
  echo "  retry:"
  echo "    automatic: true"
  echo "  agents:"
  echo "    queue: ${queue}"
}

run_mpi_pytest() {
  local test=$1
  local queue=$2
  local oneccl_env=${3:-}
  oneccl_env=$(echo ${oneccl_env//:/ })

  test_env=""
  if [[ ${queue} == *"gpu"* ]]; then
    test_env="HOROVOD_TEST_GPU=1"
  fi

  # pytests have 4x GPU use cases and require a separate queue
  run_test "${test}" "${queue}" \
    ":pytest: MPI Parallel PyTests (${test})" \
    "bash -c \"${oneccl_env} ${test_env} cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 \\\$(cat /mpirun_command) /bin/bash /pytest.sh mpi)\"" \
    5
  run_test "${test}" "${queue}" \
    ":pytest: MPI Single PyTests (${test})" \
    "bash -c \"${oneccl_env} ${test_env} cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh mpi)\"" \
    10

  run_test "${test}" "${queue}" \
    ":pytest: MPI Cluster PyTests (${test})" \
    "bash -c \"${oneccl_env} ${test_env} /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.mpi.static.xml test_static_run.py\""
}

run_mpi_integration() {
  local test=$1
  local queue=$2
  local oneccl_env=${3:-}
  oneccl_env=$(echo ${oneccl_env//:/ })

  # Run test_interactiverun.py
  if [[ ${test} != *"mpich"* ]] && [[ ${test} != *"oneccl"* ]]; then
    # TODO: support mpich
    run_test "${test}" "${queue}" \
      ":jupyter: Run PyTests test_interactiverun (${test})" \
      "bash -c \"cd /horovod/test && pytest -v --capture=no --continue-on-collection-errors --junit-xml=/artifacts/junit.mpi.integration.xml integration/test_interactiverun.py\""
  fi

  # Legacy TensorFlow tests
  if [[ ${test} != *"tf2_"* ]] && [[ ${test} != *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: MPI TensorFlow MNIST (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/tensorflow/tensorflow_mnist.py\""

    run_test "${test}" "${queue}" \
      ":tensorflow: MPI TensorFlow Eager MNIST (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/tensorflow/tensorflow_mnist_eager.py\""

    run_test "${test}" "${queue}" \
      ":tensorflow: MPI Keras MNIST (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/keras/keras_mnist_advanced.py\""
  fi

  if [[ ${test} != *"torch1_2"* ]]; then
    run_test "${test}" "${queue}" \
      ":fire: MPI PyTorch MNIST (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/pytorch/pytorch_mnist.py --data-dir /data/pytorch_datasets\""
  fi

  if [[ ${test} == *"mxnet2_"* ]] || [[ ${test} == *"mxnethead"* ]]; then
      run_test "${test}" "${queue}" \
               ":muscle: MPI MXNet2 MNIST (${test})" \
               "bash -c \"${oneccl_env} OMP_NUM_THREADS=1 \\\$(cat /mpirun_command) python /horovod/examples/mxnet/mxnet2_mnist.py\""
  else
      run_test "${test}" "${queue}" \
               ":muscle: MPI MXNet MNIST (${test})" \
               "bash -c \"${oneccl_env} OMP_NUM_THREADS=1 \\\$(cat /mpirun_command) python /horovod/examples/mxnet/mxnet_mnist.py\""
  fi

  # tests that should be executed only with the latest release since they don't test
  # a framework-specific functionality
  if [[ ${test} == *"tf1_15_0"* ]]; then
    run_test "${test}" "${queue}" \
      ":muscle: MPI Stall (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/test/integration/test_stall.py\""

    if [[ ${test} == *"openmpi"* ]] || [[ ${test} == *"oneccl"* ]]; then
      run_test "${test}" "${queue}" \
        ":terminal: MPI Horovodrun (${test})" \
        "bash -c \"${oneccl_env} horovodrun -np 2 -H localhost:2 python /horovod/examples/tensorflow/tensorflow_mnist.py\""
      run_test "${test}" "${queue}" \
        ":terminal: MPI Horovodrun (${test})" \
        "bash -c \"${oneccl_env} echo 'localhost slots=2' > hostfile && horovodrun -np 2 -hostfile hostfile python /horovod/examples/mxnet/mxnet_mnist.py\""
    fi
  fi

  # TensorFlow 2.0 tests
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: MPI TensorFlow 2.0 MNIST (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/tensorflow2/tensorflow2_mnist.py\""

    run_test "${test}" "${queue}" \
      ":tensorflow: MPI TensorFlow 2.0 Keras MNIST (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/tensorflow2/tensorflow2_keras_mnist.py\""
  fi
}

run_mpi() {
  local test=$1
  local queue=$2
  local oneccl_env=${3:-}

  run_mpi_pytest ${test} ${queue} ${oneccl_env}
  run_mpi_integration ${test} ${queue} ${oneccl_env}
}

run_gloo_pytest() {
  local test=$1
  local queue=$2

  test_env=""
  if [[ ${queue} == *"gpu"* ]]; then
    test_env="HOROVOD_TEST_GPU=1"
  fi

  run_test "${test}" "${queue}" \
    ":pytest: Gloo Parallel PyTests (${test})" \
    "bash -c \"${test_env} cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 horovodrun -np 2 -H localhost:2 --gloo /bin/bash /pytest.sh gloo)\"" \
    5
  run_test "${test}" "${queue}" \
    ":pytest: Gloo Single PyTests (${test})" \
    "bash -c \"${test_env} cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh gloo)\"" \
    15

  run_test "${test}" "${queue}" \
    ":pytest: Gloo Cluster PyTests (${test})" \
    "bash -c \"${test_env} /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.static.xml test_static_run.py\""
}

run_gloo_integration() {
  local test=$1
  local queue=$2

  # TensorFlow 2.0 tests
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: Gloo TensorFlow 2.0 MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/tensorflow2/tensorflow2_mnist.py"

    run_test "${test}" "${queue}" \
      ":tensorflow: Gloo TensorFlow 2.0 Keras MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/tensorflow2/tensorflow2_keras_mnist.py"
  else
    run_test "${test}" "${queue}" \
      ":tensorflow: Gloo TensorFlow MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/tensorflow/tensorflow_mnist.py"

    run_test "${test}" "${queue}" \
      ":tensorflow: Gloo Keras MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/keras/keras_mnist_advanced.py"
  fi

  if [[ ${test} != *"torch1_2"* ]]; then
    run_test "${test}" "${queue}" \
      ":fire: Gloo PyTorch MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/pytorch/pytorch_mnist.py --data-dir /data/pytorch_datasets"
  fi

  if [[ ${test} == *"mxnet2_"* ]] || [[ ${test} == *"mxnethead"* ]]; then
      run_test "${test}" "${queue}" \
               ":muscle: Gloo MXNet2 MNIST (${test})" \
               "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/mxnet/mxnet2_mnist.py"
  else
      run_test "${test}" "${queue}" \
               ":muscle: Gloo MXNet MNIST (${test})" \
               "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/mxnet/mxnet_mnist.py"
  fi

  # Elastic Horovod
  local elastic_tensorflow="test_elastic_tensorflow.py test_elastic_tensorflow_keras.py"
  local elastic_spark_tensorflow="test_elastic_spark_tensorflow.py"
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
      elastic_tensorflow="test_elastic_tensorflow2.py"
      elastic_spark_tensorflow="test_elastic_spark_tensorflow2.py"
  fi

  run_test "${test}" "${queue}" \
    ":factory: Elastic Tests (${test})" \
    "bash -c \"cd /horovod/test/integration && HOROVOD_LOG_LEVEL=DEBUG pytest --forked -v --log-cli-level 10 --log-cli-format '[%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d %(funcName)s()] %(message)s' --capture=no --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.elastic.xml test_elastic_torch.py ${elastic_tensorflow}\""

  # Elastic Horovod on Spark tests are very expensive (high timeout)
  # We only need to run this for our baseline test image, baseline with tensorflow1, and pyspark variations
  # *-gloo-* does intentionally not match -openmpi-gloo- here
  if [[ ${test} == ${baseline} ]] || [[ ${test} == test-cpu-gloo-*-tf1_* ]] || [[ ${test} != *pyspark3_1_2* ]]; then
    run_test "${test}" "${queue}" \
      ":factory: Elastic Spark TensorFlow Tests (${test})" \
      "bash -c \"cd /horovod/test/integration && /spark_env.sh HOROVOD_LOG_LEVEL=DEBUG pytest --forked -v --log-cli-level 10 --log-cli-format '[%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d %(funcName)s()] %(message)s' --capture=no --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.elastic.spark.tf.xml ${elastic_spark_tensorflow}\"" \
      20
  fi

  # Elastic Horovod on Spark tests are very expensive (high timeout)
  # We only need to run this for our baseline test image and pyspark variations
  if [[ ${test} == ${baseline} ]] || [[ ${test} != *pyspark3_1_2* ]]; then
    run_test "${test}" "${queue}" \
      ":factory: Elastic Spark Torch Tests (${test})" \
      "bash -c \"cd /horovod/test/integration && /spark_env.sh HOROVOD_LOG_LEVEL=DEBUG pytest --forked -v --log-cli-level 10 --log-cli-format '[%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d %(funcName)s()] %(message)s' --capture=no --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.elastic.spark.torch.xml test_elastic_spark_torch.py\"" \
      20
  fi

}

run_gloo() {
  local test=$1
  local queue=$2

  run_gloo_pytest ${test} ${queue}
  run_gloo_integration ${test} ${queue}
}

run_spark_integration() {
  local test=$1
  local queue=$2

  # Horovod Spark Estimator tests
  if [[ ${test} != *"mpich"* && ${test} != *"oneccl"* ]]; then
    if [[ ${queue} != *gpu* ]]; then
      run_test "${test}" "${queue}" \
        ":spark: Spark PyTests (${test})" \
        "bash -c \"cd /horovod/test/integration && (ls -1 test_spark*.py | xargs -n 1 /bin/bash /pytest_standalone.sh spark)\"" \
        20
    fi

    if [[ ${test} != *"tf2"* && ${test} != *"tfhead"* ]]; then
      run_test "${test}" "${queue}" \
        ":spark: Spark Keras Rossmann Run (${test})" \
        "bash -c \"OMP_NUM_THREADS=1 /spark_env.sh python /horovod/examples/spark/keras/keras_spark_rossmann_run.py --num-proc 2 --data-dir file:///data --epochs 3 --sample-rate 0.01\""

      run_test "${test}" "${queue}" \
        ":spark: Spark Keras Rossmann Estimator (${test})" \
        "bash -c \"OMP_NUM_THREADS=1 /spark_env.sh python /horovod/examples/spark/keras/keras_spark_rossmann_estimator.py --num-proc 2 --work-dir /work --data-dir file:///data --epochs 3 --sample-rate 0.01\""

      run_test "${test}" "${queue}" \
        ":spark: Spark Keras MNIST (${test})" \
        "bash -c \"OMP_NUM_THREADS=1 /spark_env.sh python /horovod/examples/spark/keras/keras_spark_mnist.py --num-proc 2 --work-dir /work --data-dir /data --epochs 3\""
    fi

    run_test "${test}" "${queue}" \
      ":spark: Spark Torch MNIST (${test})" \
      "bash -c \"OMP_NUM_THREADS=1 /spark_env.sh python /horovod/examples/spark/pytorch/pytorch_spark_mnist.py --num-proc 2 --work-dir /work --data-dir /data --epochs 3\""

    run_test "${test}" "${queue}" \
      ":spark: Spark Lightning MNIST (${test})" \
      "bash -c \"OMP_NUM_THREADS=1 /spark_env.sh python /horovod/examples/spark/pytorch/pytorch_lightning_spark_mnist.py --num-proc 2 --work-dir /work --data-dir /data --epochs 3\""
  fi
}

run_single_integration() {
  local test=$1
  local queue=$2
  local oneccl_env=${3:-}
  oneccl_env=$(echo ${oneccl_env//:/ })

  # Only in TensorFlow 1.X
  if [[ ${test} != *"tf2_"* ]] && [[ ${test} != *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: Single Keras MNIST (${test})" \
      "bash -c \"${oneccl_env} python /horovod/examples/keras/keras_mnist_advanced.py --epochs 3 --batch-size 64\""
  fi

  if [[ ${test} != *"torch1_2"* ]]; then
    run_test "${test}" "${queue}" \
      ":fire: Single PyTorch MNIST (${test})" \
      "bash -c \"${oneccl_env} python /horovod/examples/pytorch/pytorch_mnist.py --epochs 3 --data-dir /data/pytorch_datasets\""
  fi

  if [[ ${test} == *"mxnet2_"* ]] || [[ ${test} == *"mxnethead"* ]]; then
      run_test "${test}" "${queue}" \
               ":muscle: Single MXNet2 MNIST (${test})" \
               "bash -c \"${oneccl_env} python /horovod/examples/mxnet/mxnet2_mnist.py --epochs 3\""
  else
      run_test "${test}" "${queue}" \
               ":muscle: Single MXNet MNIST (${test})" \
               "bash -c \"${oneccl_env} python /horovod/examples/mxnet/mxnet_mnist.py --epochs 3\""
  fi
}

# begin the pipeline.yml file
echo "steps:"

# build every test container
for test in ${tests[@]-}; do
  build_test "${test}"
done

# wait for all builds to finish
echo "- wait"

oneccl_env="\\\$(cat:/oneccl_env):&&"
oneccl_cmd_ofi="${oneccl_env}:echo:'/mpirun_command_ofi':>:/mpirun_command:&&"
oneccl_cmd_mpi="${oneccl_env}:echo:'/mpirun_command_mpi':>:/mpirun_command:&&"

# run all the cpu unit tests and integration tests
for test in ${tests[@]-}; do
  if [[ ${test} == *-cpu-* ]]; then
    # if gloo is specified, run gloo cpu unit tests and integration tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo ${test} "cpu"
    fi

    # if oneCCL is specified, run some tests twice,
    # once with mpirun_command_ofi, and once with mpirun_command_mpi
    if [[ ${test} == *oneccl* ]]; then
      # run mpi cpu unit tests and integration tests
      run_mpi ${test} "cpu" ${oneccl_cmd_mpi}
      run_mpi ${test} "cpu" ${oneccl_cmd_ofi}

      # always run spark tests which use MPI and Gloo
      run_spark_integration ${test} "cpu"

      # no runner application, world size = 1
      run_single_integration ${test} "cpu" ${oneccl_cmd_mpi}
      run_single_integration ${test} "cpu" ${oneccl_cmd_ofi}
    else
      # run mpi cpu unit tests and integration tests
      if [[ ${test} == *mpi* ]]; then
        run_mpi ${test} "cpu"
      fi

      # always run spark tests which use MPI and Gloo
      run_spark_integration ${test} "cpu"

      # no runner application, world size = 1
      run_single_integration ${test} "cpu"
    fi
  fi
done

# wait for all cpu unit and integration tests to finish
echo "- wait"

# run 4x gpu unit tests
for test in ${tests[@]-}; do
  if [[ ${test} == *-gpu-* ]] || [[ ${test} == *-mixed-* ]]; then
    # if gloo is specified, run gloo gpu unit tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo_pytest ${test} "4x-gpu-v510"
    fi

    # if mpi is specified, run mpi gpu unit tests
    if [[ ${test} == *mpi* ]]; then
      run_mpi_pytest ${test} "4x-gpu-v510"
    fi
  fi
done

# wait for all gpu unit tests to finish
echo "- wait"

# run 2x gpu integration tests
for test in ${tests[@]-}; do
  if [[ ${test} == *-gpu-* ]] || [[ ${test} == *-mixed-* ]]; then
    # if gloo is specified, run gloo gpu integration tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo_integration ${test} "2x-gpu-v510"
    fi

    # if mpi is specified, run mpi gpu integration tests
    if [[ ${test} == *mpi* ]]; then
      run_mpi_integration ${test} "2x-gpu-v510"
    fi

    run_spark_integration ${test} "2x-gpu-v510"
  fi
done
