# Setting up oscar:

After logging in to oscar run:

```
git clone git@github.com:carlos-freire2/spatial-probing.git
```

I recommend you put these in your `~/scratch` directory (have 512GB there), it is mirrored at `/oscar/scratch/your-username` so if you see that in any of the scripts that's what it is.

Then go into the repo and run the `setup_oscar_env.sh` script. You might need to `chmod +x` it to give execute permissions. This will create a conda env named `team2d` that has all the packages we need. You might also want to `export PYTHONPATH=/path/to/sptial-probing` so that it can pickup the models module. 

## Get dataset
This is the link to the zipped dataset RAVEN-10000 ```https://drive.google.com/file/d/1fUSmWZpCsoP6sLsmqrxbnD_RO2o1zj1S/view?usp=sharing```. I recommend using `gdown` or `wget`, but you can also just download it locally and then `scp /path/to/source/file <username>@ssh.ccv.brown.edu:/path/to/destination/file` it to oscar.

After the zipped file is up run:
`unzip name_of_your_file.zip` and you'll have all 30 gigs of puzzles up there (also recommend putting this in `~/scratch`).

## Preprocess data
If you want single image embeddings, skip this step (the datasets .npz files are already split like this). 

If you want per row embeddings, run the preprocess_RAVEN.py file (assume we will add a flag saying stitch just three).

If you want per file embeddings (all 8 puzzles in one image), run the preprocess_RAVEN.py file with default args.


## Submit job
Check the slurm_embeddings.sh file to see if you are putting/outputting your files into the same dirs (if not just pass them as positional args when you sbatch it)

Run the slurm_embeddings.sh (extract_embeddings2 handles npy and npz so should not matter whether you stitched or not) using `sbatch slurm_embeddings.sh`