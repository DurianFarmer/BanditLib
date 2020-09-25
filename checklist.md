## Questions
- [x] In MLP, we matain a different network for different user? So in first version of NUCB, we do the same thing?
- [x] Indeed take days to finish yahoo dataset, so I did not check the performance on that dataset. How to split the dataset in configuration file?
- [x] In MLP snippet, the author mentioned that converting the tensor to GPU is slower than do it on CPU totally, is that true?
- [x] MLP snippet, why only 1 cpu used? I checked `torch.get_num_thread()` and set `torch.set_num_thread(8)`, but it not works.
- [x] MLP, only train with the fresh sampled data until `loss < thres`? Super overfitting?
- [x] Ln175@YahooRewardManager.py, why we only update when the guess is correct? Or I miss something important?

## TODO list
- [x] Code review
- [x] Copy the MLP version to NUCB to make code work
- [x] Revise the NUCB code on seperate user?
- [] check the lambda and weight decay scale, gradient outer scale, etc.

## Performance check
- LastFM@NUCB: 20sec@1000DP @8thread CPU, start up, 1000iter, bufsz 1024
- Linear-NeuralUCB is (slightly) weaker than LinearUCB (same parameter), might be: 1024 buffer + inv diag approx, esp. 1024 buffer on ~96k dataset.