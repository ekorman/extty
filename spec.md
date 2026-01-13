i want to support different keys for the x-axis instead of just step (e.g. maybe there is an epoch-indexed thing we want to log). this is what i'm thinking but let me know if you think this is a good pattern:

to `extty.log` add optional `step_key: str` parameter (which defaults to "step")
i think this must be added to the csv and jsonl files that get created

also we should display the step_key in the rust TUI
