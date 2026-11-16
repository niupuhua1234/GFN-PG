# Serious academic misconduct of [Proximal Policy Optimization for Amortized Discrete Sampling v2](https://arxiv.org/abs/2606.15793)

We also note that the authors have released a revised version of Proximal Policy Optimization for Amortized Discrete Sampling ([Ent-PPO](https://openreview.net/forum?id=ODbzTJmgp3)) on arXiv ([Ent-PPO v2](https://arxiv.org/abs/2606.15793)). However, changes in Ent-PPOv2  appear to make the underlying academic issues less immediately apparent rather than substantively addressing them. **We therefore maintain our serious concerns and allegations of potential academic misconduct concerning originality and attribution**. The detailed assessment is as follows.

**1.** First of all,  this work follows the Soft Actor-Critic (SAC) framwork, where the $V^\pi$ and $\pi$ are paramterized and learned seperatedly.  The value function $V^\pi$ with it learning objectives, and  advantages function $A^\pi$ for updating  $\pi$ is the same as $V_F$ and $A_F$ used in [1,2]. However, in previous version,  the author claims that they give new Soft-RL (composed of SAC [3] and Soft Q-learning (SQL) [4] ) perspective  for GFlowNet, due to difference reward interpretation for defining $V$ and $A$.
 
This updated version briefly acknowledges this equivalence with one breif sentece in "Related work p.4". Nevertheless, the main text  focus on foregrounding a connection to the SQL representation in [5,4], based on the shared reward interpretation, and **continues to emphasize a reward interpretation contrast with the earlier SAC-style formulation in [1]. Yet the proposed methodology and implementation follow the same soft actor–critic structure, with PPO used for the actor update.** This framing continues to leave readers with an overstated impression of conceptual separation from the earlier formulations in [1,2].

*  The actually acknowledgment, about the equivalence of the value function and its learning objective to [1,2], is placed in Appendix G, at the end of the updated version, and never explicitlymentioned/referred/dicussed in the main text. This selective presentation obscures the extent of the methodological overlap and creates a misleading impression of conceptual novelty.

*  In contrast, a substantial portion of the main text is devoted to presenting and explaining the consequences of the following formulation, 
   $V^\pi(s):=E_\pi\left[\sum_{t=0}^{T-1}\log \pi_B(s_t|s_{t+1})-E_{s_{t+1}}[\log \pi(s_{t+1}|s_t)]\mid s_0=s\right]=E_\pi\left[\sum_{t=0}^{T-1}\log \frac{\pi_B(s_t|s_{t+1})}{\pi(s_{t+1}|s_t))}|s_0=s\right]$,By interpreting $\log\pi_B(s\mid s')$ as the intermediate reward $r(s,s')$ and writing the entropy $H(s_t)=E_{s_{t+1}}[\log \pi(s_{t+1}|s_t)]$ explicitly, the authors suggest methodological consequences different from those associated with the corresponding equivalent formulation in [1,2].

* By continuing to emphasize the above reward interpretation, the authors aims at redirecting readers’ attention from the method’s SAC structure to the SQL structure in [4,5] (a another parallel Soft-RL branch), which also interprets $\log\pi_B$ as the intermediate reward.  With substantial main-text discussion devoted to foregrounding this connection, while the fuller correspondence with [1,2,3] is deferred to Appendix G, the updated version sidesteps a detailed comparison with the most directly relevant earlier formulations.

   To be specific, 

   | | SAC-style actor–critic | Soft Q-learning (SQL) |
   |---|---|---|
   | **What is optimized?** | Critic $\widehat V$ and actor $\pi$, using separate objectives | $\hat Q$, or its parameterization $\hat Q=\widehat V+\log\pi$ |
   | **Value-learning target** | The current policy’s value: $V^\pi(s)=\log F^\ast(s)-D_{KL}(P_\pi(\tau\mid s)\| P^\ast(\tau \mid s))$ | The optimal value: $Q^\ast(s,s')=\log F^\ast(s,s')$ or $\log F^\ast(s)+\log \pi^\ast(s'\mid s)$ |
   | **Policy update** | Improve the actor using the critic | Derive the policy from the learned $Q$ |
   | **At the joint optimum** | $\widehat V=V^\ast=\log F^\ast$, $\pi=\pi^\ast$ | $\widehat Q=Q^\ast=V^\ast+\log\pi^\ast$ |

  Table 1 of this updated version shows the shared optimal quantities, but does not explain this difference in learning targets. The updated version trains a separate critic to estimate $V^\pi$ and uses it to improve the actor. This follows the actor–critic structure in [1,2,3], with PPO  [6] supplying the actor update. The shared optimal-value correspondence therefore does not establish a distinct conceptual framework.


**2.**  As motivations of their work, the authors introduce two related concerns about an allegedly omitted entropy correction and multi-epoch updates:
   > Introduction, p. 2: "the standard practice of reusing advantages computed at the rollout policy silently drops an entropy correction (Section G)."

   > Related Work, p. 4:  “with a policy-dependent reward the standard practice of reusing rollout advantages drops this term”. 

   Here, “this term” refers to the unit-weight KL correction.

   > Introduction,  p. 2: “This formulation has the same objective and on-policy gradient as the entropy-regularized one, but in multi-epoch updates such as PPO the reward changes together with the policy, and the standard practice of reusing advantages computed at the rollout policy silently drops an entropy correction (Section G).”

   > Related Work, p. 4: “They differ in multi-epoch updates: in the stationary formulation, the correct off-policy surrogate, including a KL term of unit weight, is the default, whereas with a policy-dependent reward the standard practice of reusing rollout advantages drops this term”

 **These issues arise from the authors’ own surrogate formulation of the performance-difference lemma in [7,8] and the soft version in [1], rather than from the surrogate formulations adopted in [1,7,8]**. Nevertheless, the authors use the weaknesses of their own construction to criticize the earlier formulations.

* Let $p=\pi_{\mathrm{old}}$ and $q=\pi_\theta$, 

  | Quantity | Earlier formulation in [1,2] | Correspondence in this updated version |
  |---|---|---|
  | **Policy value** | $V_F^p(s)$ | $V^{\pi_o}(s)=-V_F^p(s)$ |
  | **Advantage** |  $A_F^p(s,a)$ | $A^p(s,a)=-A_F^p(s,a)$ |
  | **Exact performance difference** |  Lemma C.2: $J_F(q)-J_F(p)= \sum_t \mathbb E_{s_t\sim d^q} \left[ \mathbb E_{a_t\sim q} A_F^p(s_t,a_t)+KL(q\|p)(s_t)\right]$ |  Eq. (24):  $J(q)-J(p)=\mathbb E_{\tau\sim q}\sum_t \left[A^p(s_t,a_t)-\mathrm{KL}(q\|p)(s_t)\right]$ $\quad\quad = \sum_t \mathbb E_{s_t\sim d^q} \left[ \mathbb E_{a_t\sim q} A^p(s_t,a_t)-\mathrm{KL}(q\|p)(s_t)\right]$ |
  | **Local Soft improvement** |  $-\mathbb E_q[A_F^p(s,a)]-\mathrm{KL}(q\|p)(s)$ | Eqs. (15)–(17): $\mathbb E_q[A^p(s,a)]-\mathrm{KL}(q\|p)(s)$ |
  | **Practical soft objective choice** |  Unconstained soft-TRPO objective [1,8]:  $\mathbb E_{s\sim d^p}[-\mathbb E_{a\sim q}[A_F^p(s,a)]-\beta\mathrm{KL}(p\|q)(s)]$ | Eqs. (15)–(17):   $\mathbb E_{s\sim d^p}[-\mathbb E_{a\sim q}[A_F^p(s,a)]-\textcolor{red}{\mathrm{KL}(q\|p)(s)}]$ |

 * The corresponding formations in vanilla TRPO [7,8], take the same forms as the first column above,  only differing at the intermediate reward is policy-independent.  Therefore, the author  not only attributes the two issues with Soft-TRPO [1], but also the vanilla TRPO [7,8].

   Accoding to the table, both [1,7,8] and this updated version use fixed rollout advantages $A^p$, the rollout-policy state distribution induced by $p$, and explicit unit-KL regularization.  Therefore, the omiited correction claimed by this updated version is unreasonable.



* The claimed multi-epoch issues arise from the $\textcolor{red}{\mathrm{KL}(q\|p)(s)}$  of the soft-objective in updated version, rather than in [1,7].  Here, $K$ update epochs correspond to $K$ actor gradient-updates on the rollout trajectory batch collected in one training iteration, with advantages held fixed throughout (Appendix D, Algorithm 2, lines 2–7, p. 23).
  
   *  In particular, the earlier surrogate uses $\mathrm{KL}(p\|q)(s)=\mathbb E_{a\sim p}[\log p(a\mid s)-\log q(a\mid s)]$, which can be unbiasedly esitimated  by $\hat KL(p\|q)(s) =\frac{1}{|\mathcal B|}\sum_{(s,a) \in \mathcal B} \log p(a\mid s)-\log q(a\mid s)$, where $\mathcal B$ is the rollout batch sampled from $p$. The batch $\mathcal B$  remain valid throughout all $K$ updates, as $\mathbb E[\hat KL(p\|q)(s)]=\mathrm{KL}(p\|q)(s)$. Consequently, changing $q$ provides no basis for attributing a rollout-reuse problem to this KL term.

   *   The updated version instead uses $\textcolor{red}{\mathrm{KL}(q\|p)(s)}$. An unweighted average over actions from the original rollout batch would not correctly estimate this quantity once $q\neq p$. This direction requires analytical computation, importance weighting, or samples from the updated $q$. The authors use analytical computation, and it does not establish a deficiency in the earlier formulation, whose KL term can already be estimated directly from reused rollout samples.


3. For the soft performance-difference lemma (PDL), on which Ent-PPO is built, the authors previously stated:

   >  equivalent statements appear in Neu et al.  and Geist et al..
 
   Now in this updated version, the author states that  

     >  equivalent statements appear in Neu et al.  and Geist et al. , and in the GFlowNet setting it also appears in the proof of Lemma C.2 of Niu et al. .
    
    After examining Neu et al. and Geist et al., I could not locate the exact correspondance stated in those works. By contrast, the identity appears explicitly in the proof of Lemma C.2 of Niu et al. (that is, [1] in this OpenReview discussion). **Therefore, the author still fail to faithfully attribute contribution to prior works**. 


4.  For Proposition 5, its proof use the occupancy-mismatch bound  ( Lemma B.5 of [1] ) for absorbing MDPs on finite-horizon DAGs. However, the authors refer to Achiam et al.’s Lemma 3 [8], which concerns discounted regular MDPs. The occupancy definitions and proof representations differ between the two results. **Therefore, the author fail to faithfully attribute contribution to prior works again**.

# Reference 


[1] Niu, P., Wu, S., Fan, M., and Qian, X. (2024). GFlowNet training by policy gradients. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 38344–38380. PMLR.

[2] Niu, P., Wu, S., and Qian, X. (2026). Evaluating GFlownet from partial episodes for stable and flexible policy-based training. In The Fourteenth International Conference on Learning Representations.

[3] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861–1870. PMLR, 2018.

[4] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In International conference on machine learning, pp. 1352–1361. PMLR, 2017.


[5] Daniil Tiapkin, Morozov Nikita, Naumov Alexey, and Vetrov P Dmitry. Generative flow networks as entropy-regularized rl. In International Conference on Artificial Intelligence and Statistics. PMLR, 2024.

[6] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017b). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

[7] Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. (2015a). Trust region policy optimization. In International Conference on Machine Learning, pages 1889–1897. PMLR.

[8] Achiam, J., Held, D., Tamar, A., and Abbeel, P. Constrained policy optimization. In International conference on machine learning, pp. 22–31. PMLR, 2017.
