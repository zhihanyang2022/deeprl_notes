# Deep RL: Notes on Policy Gradient

Zhihan Yang @ June 19, 2020

Note that we are dealing with finite episodes here.

## Policy gradient

**Goal.** We seek to maximize the following quantity:
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta(\tau)} \left[ \sum_{t=1}^T r(s_t, a_t) \right]
$$

$$asd$$ hahahah