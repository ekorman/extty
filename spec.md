spec for a new machine learning tracker offering. Similar to tensorboard or weights and biases. strictly focused on metrics logging (such as in tensorboard) and not interested in artifacts or workflows.

it will have a very nice terminal UI, which is useful when in dev mode (don't want to bother logging things to a backend server) and also when running code remotely (where its annoying to pass API creds, or spin up a tensorboard instance and then port-forward/ssh tunnel etc)

# components

three components:

## python library

the python library `extty` will have an API that is similar to `tensorboardX` or `wandb`. where the pattern is

1. init a project with something like

```py
extty.init("project-name", conf={"hyperparameter-1": 1.7, "hyperparameter-2": True})
```

2. log various metrics

```py
extty.log({"train/loss": 0.0192}, step=110)
extty.log({"val/f1": 0.85}, step=110)
extty.log({"val/example": {"prompt": "what is the capital of france", "response": "paris"}}, step=200)
```

3. also capture system information such as memory (both RAM and VRAM), steps per second, etc.

## terminal UI

there is a single binary file `extty` that is a beautiful terminal user interface (maybe using the `ink` typescript library). this allows displaying both live metrics for active training runs but also locally existing ones. this should be able to display line charts, examples (at least for text tasks, maybe not image tasks), system stats.

## a web backend and web interface

a web app reminiscent of tensorboard or weights and biases. it will have centralized experiment data to faciliate comparison (e.g. compare results of training with different hyperparemters). everything will be very structured and queryable so that e.g. you can compare across datasets or models. at some point would like the ability to dynamically create graphs from a chatbot, where .e.g. the user says "give me data visualizations comparing all of the training runs for model architecture X on dataset Y"

# things to work out

- data format?
- how to sync data from local to remote?
