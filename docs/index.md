# Deep RL: Notes on Policy Gradient

Zhihan Yang @ June 19, 2020

Note that we are dealing with finite episodes here.

## Policy gradient

**Goal.** We seek to maximize the following quantity:
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta(\tau)} \left[ \sum_{t=1}^T r(s_t, a_t) \right]
$$
where:

- $$\theta$$ is the parameters of a neural network.
- $$\tau = \{s_1, a_1, s_2, a_2, \cdots, s_{T-1}, a_{T-1}, s_T \}$$
- $$\pi_{\theta}(\tau) = p(s_1) \prod_{t=1}^{T-1} \pi_\theta(a_t | s_t) p(s_{t+1}|s_t, a_t)$$
- $$r$$ is the reward function (built-in to the environment).

**Idea.** $$\theta_{\text{new}} = \theta_{\text{old}} + \nabla_{\theta} J(\theta)$$, now we need to consider how to evaluate $$\nabla_{\theta} J(\theta)$$.

**Derivation.** Here we derive an easy-to-evaluate form of $$\nabla_{\theta} J(\theta)$$. 

- For the sake of unclustered notation, we use $$r(\tau) = \sum_{t=1}^T r(s_t, a_t)$$.

- Handy identities
    - Identity 1: $$\nabla_v f(v) = f(v) \frac{\nabla_v f(v)}{f(v)}=f(v) \nabla_v \log(f(v))$$

$$
\begin{align*}
\nabla_{\theta} J(\theta) 

&= \nabla_\theta \left\{ \mathbb{E}_{\tau \sim \pi_\theta(\tau)} \left[ r(\tau) \right] \right\} \\

&= \nabla_\theta \left\{ \int \pi_\theta(\tau) r(\tau) d\tau \right\} \\

&= \int \nabla_\theta \left\{ \pi_\theta(\tau) \right\} r(\tau) d\tau\\

&= \int \pi_\theta(\tau) \nabla_\theta \left\{ \log(\pi_\theta(\tau)) \right\} r(\tau) d\tau \tag*{By identity 1} \\

&= \mathbb{E}_{\tau \sim \pi_{\theta}(\tau)} \left[ \nabla_\theta \left\{ \log(\pi_\theta(\tau)) \right\} r(\tau) \right] \\

&= \mathbb{E}_{\tau \sim \pi_{\theta}(\tau)} \left[ \nabla_\theta \left\{ \log p(s_1) + \sum_{t=1}^{T-1} \log(\pi_\theta(a_t \mid s_t)) + \sum_{t=1}^{T-1} \log(p(s_{t+1} \mid s_t, a_t)) \right\} r(\tau) \right] \tag*{By definition of $\pi_\theta(\tau)$}\\

&= \mathbb{E}_{\tau \sim \pi_{\theta}(\tau)} \left[ \nabla_\theta \left\{\sum_{t=1}^{T-1} \log(\pi_\theta(a_t \mid s_t))\right\} r(\tau) \right] \tag*{Cancelled irrelevant terms}\\

&= 
\mathbb{E}_{\tau \sim \pi_{\theta}(\tau)} \left[ 
\underbrace{
\left( \sum_{t=1}^{T-1} \nabla_\theta \left\{ \log(\pi_\theta(a_t \mid s_t)) \right\} \right) 
}_{\text{gradient in favor of } \tau}
\underbrace{r(\tau)}_{\text{ reward of } \tau} 
\right] \\

\end{align*}
$$

In practice, this expectation can be evaluated simply by sampling trajectories.

We can interpret the purpose of this gradient as increasing the probability of high reward trajectories and decreasing the probability of low reward trajectories.

## Comparison to supervised learning (maximum likelihood)

We seek to maximize the following quantity using the maximum likelihood approach:
$$
J_{\text{ML}}(\theta)=\mathbb{E}_{\tau \sim p_{\text{train}}(\tau)}\left[\sum_{t=1}^T \log \pi_{\theta} (a_t | s_t)\right]
$$
Note that the expectation is over trajectories sampled from the training distribution, not the on-policy distribution.

The easy-to-evaluate form of its gradient can be derived as follows: 
$$
\begin{align*}
\nabla_{\theta} \left\{ J_{\text{ML}}(\theta) \right\} &=\nabla_{\theta} \left\{ \mathbb{E}_{\tau \sim p_{\text{train}}(\tau)}\left[\sum_{t=1}^T \log p_{\theta} (a_t | s_t)\right]  \right\} \\

&= \nabla_{\theta} \left\{ \int p_{\text{train}}(\tau) \left[\sum_{t=1}^T \log p_{\theta} (a_t | s_t)\right] d\tau \right\} \\

&= \int  p_{\text{train}}(\tau) \nabla_{\theta} \left\{ \sum_{t=1}^T \log p_{\theta} (a_t | s_t)\right\} d\tau\\

&= \int  p_{\text{train}}(\tau) \left\{ \sum_{t=1}^T  \nabla_{\theta}\log p_{\theta} (a_t | s_t)\right\} d\tau\\

&= \mathbb{E}_{\tau \sim p_{\text{train}}(\tau)} \left[ 
\underbrace{\sum_{t=1}^T  \nabla_{\theta}\log p_{\theta} (a_t | s_t)}_{\text{gradient in favor of }\tau} 
\right]

\end{align*}
$$
The differences between behavior cloning and vanilla policy gradient are summarized below:

|        Method        |                       Policy gradient                        |            Maximum likelihood (behavior cloning)             |
| :------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Function to maximize | $$J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta(\tau)} \left[ \sum_{t=1}^T r(s_t, a_t) \right]$$ | $$J_{\text{ML}}(\theta) = \mathbb{E}_{\tau \sim p_{\text{train}}(\tau)}\left[\sum_{t=1}^T \log \pi_{\theta} (a_t, s_t)\right]$$ |
|       Gradient       | $$\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}(\tau)} \left[ \underbrace{\left( \sum_{t=1}^{T-1} \nabla_\theta \left\{ \log(\pi_\theta(a_t \mid s_t)) \right\} \right) }_{\text{gradient in favor of } \tau}\underbrace{r(\tau)}_{\text{ reward of } \tau} \right]$$ | $$\nabla_{\theta}J_{\text{ML}}(\theta) = \mathbb{E}_{\tau \sim p_{\text{train}}(\tau)} \left[ \sum_{t=1}^T \nabla_{\theta}\log p_{\theta} (a_t, s_t) \right]$$ |
| MC gradient estimate | $$\nabla_{\theta}J(\theta) \approx \frac{1}{N} \sum_{n=1}^N \left\{ \underbrace{\left( \sum_{t=1}^{T-1} \nabla_\theta \left\{ \log(\pi_\theta(a_{n, t} \mid s_{n, t})) \right\} \right) }_{\text{gradient in favor of } \tau_n}\underbrace{r(\tau_n)}_{\text{ reward of } \tau_n} \right\}$$ | $$\nabla_{\theta}J_{\text{ML}}(\theta) \approx \frac{1}{N} \sum_{n=1}^N \left\{ \underbrace{\sum_{t=1}^T \nabla_{\theta}\log p_{\theta} (a_{n, t} \mid s_{n, t})}_{\text{gradient in favor of } \tau_n} \right\}$$ |

Notes:

- In English, $$\nabla_{\theta} J(\theta)$$ weights the gradient in favor of $$\tau_n$$ by its reward, while $$\nabla_{\theta}J_{\text{ML}}(\theta)$$ weights all gradients equally. 
- The similarity between two gradients will help us compute the policy gradient, as we will see later.

## REINFORCE algorithm (vanilla version)

- Initialize $$\theta$$. 

- Loop:
    - Sample $$\{\tau_n\}$$ by running $$\pi_{\theta}(a_t|s_t)$$ in some environment.
    - Compute $$\nabla_{\theta}J(\theta)$$ by using its MC gradient estimate rule in the table above.
    - $$\theta_{\text{new}} \leftarrow \theta_{\text{old}} + \alpha \nabla_{\theta}J(\theta)$$. (A more advanced optimizer can be used in practice.)

## Problems with vanilla REINFORCE and its variants (remedies)

Reference: answer by Jerry Liu: https://www.quora.com/Why-does-the-policy-gradient-method-have-a-high-variance

### Reward-to-go (exploiting causality)

Since we will be estimating policy gradients by sampling trajectories, the variance of the resulting gradients can be high. To reduce this variance, we need to eliminate as many random variables as possible from the MC gradient estimate formula. To do so, we first re-write the MC gradient estimate formula as follows:
$$
\begin{align*}
\nabla_{\theta}J(\theta) 

&\approx \frac{1}{N} \sum_{n=1}^N \left\{ \left( \sum_{t=1}^{T-1} \nabla_\theta \left\{ \log(\pi_\theta(a_{n, t} \mid s_{n, t})) \right\} \right) \left( \sum_{t=1}^{T-1} r(s_{n, t}, a_{n, t}) \right)\right\} \\

&= \frac{1}{N} \sum_{n=1}^N \left\{ \left( \sum_{t=1}^{T-1} \nabla_\theta \left\{ \log(\pi_\theta(a_{n, t} \mid s_{n, t})) \right\} \left( \sum_{t'=1}^{T-1} r(s_{n, t'}, a_{n, t'}) \right)\right) \right\} \\

&= \frac{1}{N} \sum_{n=1}^N \left\{ \left( \sum_{t=1}^{T-1} \nabla_\theta \left\{ \log(\pi_\theta(a_{n, t} \mid s_{n, t})) \right\} \left( \sum_{t'=t}^{T-1} r(s_{n, t'}, a_{n, t'}) \right)\right) \right\} \\

\end{align*}
$$
where we exploited causality (future actions do not impact past rewards) in the last step to remove all $$r(s_{n, t’}, a_{n, t’})$$ where $$t’ < t$$. 

To see why each reward is a random variable, consider the reward of the $$k$$-th state on the $$i$$-th sampled trajectory. Obviously, this reward, $$r(s_{i, k}, a_{i, k})$$ can be a random variable depending on the outcomes of following random processes:

- Randomness in reward (aka. the reward function itself maybe stochastic).
- Randomness in what $$s_{i, k}$$ is.
- Randomness in what $$a_{i, k}$$ is given $$s_{i, k}$$.

 ### Baseline (normalizing rewards)

To start off, we re-write the MC gradient estimate formula as follows:
$$
\begin{align*}
\nabla_{\theta}J(\theta) 

&\approx \frac{1}{N} \sum_{n=1}^N \left\{ \left( \sum_{t=1}^{T-1} \nabla_\theta \left\{ \log(\pi_\theta(a_{n, t} \mid s_{n, t})) \right\} \right) r(\tau_n) \right\} \\

&\approx \frac{1}{N} \sum_{n=1}^N \left\{ \nabla_{\theta}\{ \log \pi_{\theta} (\tau_n)\} r(\tau_n) \right\} \\

\end{align*}
$$
where $$\nabla_{\theta}\{ \log \pi_{\theta} (\tau_n)\}$$ and $$r(\tau_n)$$ are both random variables.

For two uncorrelated random variables, the variance of their product can be re-written as:
$$
\begin{align}
\text{Var}(X, Y) 
= (\sigma_{X}^2 + \mu_{X}^2)(\sigma_{Y}^2 + \mu_{Y}^2) - \mu_X^2\mu_Y^2
\end{align}
$$
If $$Y$$ is demeaned, we have
$$
\text{Var}(X, Y) = (\sigma_{X}^2 + \mu_{X}^2)(0 + \mu_{Y}^2) - \mu_X^2\mu_Y^2
$$

