# Behavior cloning

Behavior cloning refers to learning the policy from expert-provided observation-action pairs via supervised learning. Here, we compare the performance of the following two approachs

- Approach 1: a neural-net regressor and the mean-squared error (MSE)
- Approach 2: a neural-net parameter estimator, a Gaussian mixture model (GMM) and the negative log-likelihood (NLL)

on a multi-modal data-set. Multi-modality refers to the fact that the distribution of actions conditioned on observations can be multi-modal at times. 

## Data-set

I created a data-set and named it the pseudo-driving data-set (multi-modal version). The following figure is the driving track, a rectangular binary array in which ones represent the track and zeros represent off-the-track regions. The transition point between the “sine function” and the “fork” is the point of multi-modality. Note that this figure has been transposed for the purpose of display; in my code, this longer axis of this array goes vertically. 

<img src="https://i.loli.net/2020/07/07/mSF1bNKDcnV79Xy.png">

Here’s an example observation (10 rows, 40 columns); its corresponding action is [5, 9], which means that the expert moved 5 rows downwards and 9 columns to the right. In training, the first number “5” is ignored because it’s the step-size and thus is the same every time-step.

![download (2)](https://i.loli.net/2020/07/07/vlOM6A7fBc2FCoQ.png)

Here’s the track labelled by expert actions (the pink dots). Expert actions do not actually show up in observations in training data; the following figure is just for visualization. 

![download (3)](https://i.loli.net/2020/07/08/9mvAUwIK1k8PDBM.png)

This data-set is created using `./notebooks/create_pseudo_driving_dataset_multimodal.ipynb` saved as `./data/pseudo_driving_dataset_multimodal.json`.

To load it as two numpy arrays of observations and labels:

```python
with open('pseudo_driving_dataset_multimodal.json', 'r') as json_f:
    states, actions = map(np.array, json.load(json_f))
    
states = states.reshape(-1, 10 * 40)  # flatten, to be fed into a fully-connected network as 1-d vectors
actions = actions[:,1].reshape(-1, 1)  # only take the delta_y's, since delta_x's are fixed 

print(states.shape, actions.shape)
# output: (226, 400) (226, 1)
print(states.min(), states.max(), actions.min(), actions.max())
# output: 0.0 1.0 -10 10 ; actions need no normalization because we are doing regression here
```

We use no validation data because we are just trying to minic the experts’ behavior as accurate as possible. 

## Approach 1

Agent’s trajectory:

![multimodal_regressor_agent_trajectory](https://i.loli.net/2020/07/08/ofKdHwpGgy1DuT9.png)

Clearly, at the point of multi-modality, MSE loss caused the agent to go out of track. Fortunately, the agent was able to self-correct after that and return back to the track, indicating that the agent was able to generalize a little.

From more details, see `./notebooks/behavior_cloning_on_pseudo_driving_dataset_multimodal_regressor.ipynb`.

## Approach 2

Agent’s trajectory:

![multimodal_gmm_agent_trajectory](https://i.loli.net/2020/07/08/oVFpGfxN8smnlw1.png)

If you sample enough trajectories, you will see two different trajectories that are both valid.

From more details, see ``./notebooks/behavior_cloning_on_pseudo_driving_dataset_multimodal_gmm.ipynb``. 