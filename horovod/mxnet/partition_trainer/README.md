
# partition trainer

This is mxnet version of [***ZeRO: Memory Optimizations Toward Training Trillion
Parameter Models***](https://arxiv.org/pdf/1910.02054.pdf)

Currently, we implement the first stage: partition optimizer states. The utilization is easy
and same as original trainer:

```bash
#trainer = hvd.DistributedTrainer(target_params, args.optimizer, optimizer_params)
trainer = POS_Trainer(target_params, args.optimizer, optimizer_params)
trainer.allreduce_grads()
trainer.update(1.0)
```

And the saving memory is about ***K * N * (K-1)/ P***(corresponding to the parameter partition), where K is the memory
multiplier of optimizer states(12 in adam), N is the parameter number, P is the DP degree.

By using this, we can increase our batch size. Here are some improvement when using gluonnlp 
pretrained model scirpt in 8 V100 with 16GB memory. (with seq length 128)

| model    | maximum batch size with original trainer | maximum batch size with Pos trainer |
|---------|---------------------------------|---------------------------|
| bert-large | 16 | 24 |
| bert-base | 64 | 80 |


