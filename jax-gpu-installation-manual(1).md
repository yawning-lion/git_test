未名一号高性能计算器jax安装指北

1. ssh [user_id]@162.105.133.134

type in password to log in the account on cluster 

2. module load anaconda/3.7.1

this is to load anaconda3

3. conda create --name jax-gpu python=3.7.1 matplotlib

create the environment called "jax-gpu"

4. source activate jax-gpu

5. pip install jax jaxlib==0.1.74+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_releases.html

in this step, you need to check the versions of cuda and cudnn and make sure that the jaxlib has the correct version that is compatible with that of cuda and cudnn. for example in weiming-no.1 cluster, cuda has version 11.1 and cudnn 8.0-v5, and therefore, we chose to use the above package specs.
WARNING: the jax is by default the latest version and therefore can conflict with previous jaxlib versions. One needs to make sure that this does not happen by specifying the jax version as well.

6. if you see "Successfully installed absl-py-1.0.0 flatbuffers-2.0 jax-0.2.26 jaxlib-0.1.74+cuda11.cudnn805 opt-einsum-3.3.0 scipy-1.7.3 typing-extensions-4.0.1" then it means that the package has now been successfully installed.

#################################################

performance test on the gpu node

to log in onto the gpu node, see https://hpc.pku.edu.cn/guide_tensorflow.html

1. salloc --gres=gpu:1 -p GPU

it will show the gpu node that is assigned to you. suppose this node is "gpu08"

2. ssh gpu08

type in your password to log in

3. module load anaconda/3.7.1

4. source activate jax-gpu

5. python test_fft_jax.py

assuming you have already transferred the file "test_fft_jax.py" to the cluster. and you will see a speed up of around 100, just to simply replace numpy.fft.fft to jit(jnp.fft.fft) for a 1024 * 1024 array.


