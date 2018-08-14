# Restore

> There exist at least two situations where we want to restore model.
> One is when we have to resume training, the other is when we want to evaluate the model performance or generate testing results.
> Here, we use _resume training_ as an example to illustrate how to restore model checkpoint.

Note that the script does not save the graph of tensorflow. But we use the python code of the model to indicate how outputs are computed.