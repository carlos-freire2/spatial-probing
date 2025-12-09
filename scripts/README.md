# Setting up oscar:

After logging in to oscar run:

```
git clone git@github.com:carlos-freire2/spatial-probing.git
```

I recommend you put these in your `~/scratch` directory (have 512GB there), it is mirrored at `/oscar/scratch/your-username` so if you see that in any of the scripts that's what it is.

Then go into the repo and run the `setup_oscar_env.sh` script. You might need to `chmod +x` it to give execute permissions. This will create a conda env named `team2d` that has all the packages we need. You might also want to `export PYTHONPATH=/path/to/sptial-probing` so that it can pickup the models module (this is already being handled for batch jobs). 

## Get dataset
This is the link to the zipped dataset RAVEN-10000 ```https://drive.google.com/file/d/1fUSmWZpCsoP6sLsmqrxbnD_RO2o1zj1S/view?usp=sharing```. I recommend using `gdown` or `wget`, but you can also just download it locally and then `scp /path/to/source/file <username>@ssh.ccv.brown.edu:/path/to/destination/file` it to oscar.

After the zipped file is up run:
`unzip name_of_your_file.zip` and you'll have all 30 gigs of puzzles up there (also recommend putting this in `~/scratch`).

## Preprocess data

**Run these preprocess commands in an interact shell**


`interact -n 8 -t 01:00:00 -m 10g` will give you 10gb of memory and 8 cores– was enough for me to run them

If you want single image embeddings, skip this step (the datasets .npz files are already split like this). 

If you want per row embeddings, run the preprocess_raven_rows.py file.

If you want per file embeddings (all 8 puzzles in one image), run the preprocess_RAVEN.py file with default args.


## Submit job
Check the slurm_embeddings.sh file to see if you are putting/outputting your files into the same dirs (if not just pass them as positional args when you sbatch it)

Run the slurm_embeddings.sh (extract_embeddings_single handles npy and npz so should not matter whether you stitched or not but the default is set to running extract_embeddings which handles stitched images specifically) using `sbatch slurm_embeddings.sh`

If you get an OOM error, please let me know (hopefully they are all fixed already 🤷) and try running it per subdirectory at a time instead.


> DINOv3 Auth Note: 
> You will need to be authenticated to access the DINO models. Make a huggingface.co account and then go to `https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m`. There will be a form to fill out and you will basically immediately get access. Grab
> an access token or ssh key (whatever process you are more familiar with/like more) for oscar and you should be all set.
> I prefer using `export HF_TOKEN=my-token` but you can use `huggingface-cli login` or whatever other method you want.

**I'm sure y'all know this, but please do not commit any tokens to the repo**

## Running the models
Once you have your embedding files go ahead and run `sbatch slurm_run_experiment.sh` you may need to tweak the file to match your directory structure, but otherwise should be smooth sailing.