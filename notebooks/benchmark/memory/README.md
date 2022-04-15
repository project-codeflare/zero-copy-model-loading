# notebooks/benchmark/memory

This directory contains short Python scripts that we used to test the memory footprint
of loading the four models in our benchmark study into local process memory.

To run these scripts and capture peak heap size, use the `/usr/bin/time` command, 
which is *different* from your shell's built-in `time` command.

On Linux:
```
/usr/bin/time -v python3 <script>.py
```

On MacOS:
```
/usr/bin/time -l python3 <script>.py
```
Look for "maximum resident set size" or something similar in the output.

Peak heap sizes in bytes from running these scripts on my Mac:

* `intent.py`: 2349740032
* `sentiment.py`: 1159647232
* `qa.py`: 1189236736
* `generate.py`: 1391460352