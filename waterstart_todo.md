# Ideas and TODOs

- subtraction of initial price
- make the nn choose the size of the position. The loss needs to be updated too.
- turn the two main functions in classes so that we can store the state variable (cum_prob, last_prob, hx..)
- can we make the chunk size dynamic? possible ideas:
  - in the loop double the chunk size until all probs converged, up to a certain maximum.
  - make the nn predict the number of steps necessary.
  - use the last chunk size for the next iteration.
- print the computation graph with torchviz and check that everything is alright
- sometimes the cuDNN error: CUDNN_STATUS_EXECUTION_FAILED happened. We still don't know why