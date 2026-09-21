# thesis-code

code from my thesis

## thesis, for reference

[thesis.pdf](thesis/thesis_Winiarski_(watermarked).pdf)

## how to run (recommended)

```bash
# set up venv (once)
python -m venv venv
# activate venv
source venv/bin/activate # Linux
./venv/Scripts/activate # Windows
# initialise requirements (once)
pip install -r requirements.txt
# run
python <name>.py
```

## how to generate plots

in progress...

### Moir\'e deflectometer

#### Figure 3.5

![fig35](moire/particles_amin_heatmap.png)
`python moire/particles_heatmap.py` 
then wait for the simulation to run, or press Ctrl+C to load the data from CSV

### Mach--Zehnder interferometer

#### Figure 3.8

![fig38](machzehnder/fit_100g_unwrapped.png)

#### Figure 3.10

![fig310](machzehnder/parallel_amin_heatmap.png)
`python machzehnder/optimize_velocity.py`
then wait for the simulation to run, or press Ctrl+C to load the data from CSV
