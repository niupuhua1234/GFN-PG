# Public Note on the Chronology and Technical Relationship Between GFN-PG (Soft-TRPO) and Ent-PPO

This GitHub document serves as a persistent mirror of the **[Original OpenReview technical comparison](https://openreview.net/forum?id=k2lkeCCfRK&noteId=R9ob5DJqcf)**. Substantive updates can be tracked through the Git commit history.


## 1. Technical Comparison and Originality/Attribution Concern


This OpenReview page records the earliest publicly available version of [1] ([GFlowNet training by policy gradients](https://openreview.net/forum?id=G1igwiBBUj), GFN-PG), made public on October 13, 2023. [1] was subsequently accepted at ICML 2024, and [2] ([Evaluating GFlownet from partial episodes for stable and flexible policy-based training](https://openreview.net/forum?id=kWM0etSpBG), Sub-EB) appeared at ICLR 2026. This comment documents the technical relationship between our works [1,2] and the paper [9] ([Proximal Policy Optimization for Amortized Discrete Sampling](https://openreview.net/forum?id=ODbzTJmgp3). The comparison is based solely on publicly available versions of these papers.


As detailed below, the issues we identify go substantially beyond ordinary overlap in topic or an incomplete related-work discussion.

* The value and advantage formulation, the Soft-RL interpretation, the policy-optimization objective, the performance-difference result, and the performance-bound derivation presented in [9] all have direct or closely related counterparts in [1,2]. Most importantly, the central Ent-PPO objective can be obtained directly by applying the standard PPO approximation in [6] to the Soft-TRPO objective introduced in [1], while key supporting theoretical results either coincide with or closely follow results already established in [1].

* Nevertheless, [9] presents several of these elements as its own conceptual, methodological, and theoretical contributions without
clearly distinguishing the previously established results from its genuinely new contributions. 

**We therefore consider this to be a serious originality and attribution problem, rather than merely a matter of missing citations.**
The detailed technical evidence supporting these statements is provided below.


**Summary of Technical Correspondences**
| Aspect | Our earlier work | Subsequent work | Technical relationship |
|---|---|---|---|
| **Value and advantage functions** | Our earlier policy-gradient formulation [1,2] defines the soft value function $V_F$ and advantage function $A_F$ for GFlowNet. | The subsequent work introduces value $V^\pi$ and advantage $A^\pi$. | In the GFlowNet setting, after aligning notation, $V^\pi=-V_F$ and $A^\pi=-A_F$. Besides, [9] use the same objectives, TD-$\lambda$ adopted in [1] or Sub-EB[2] for learning $V^\pi$. Yet [9] presents its formulation as avoiding a non-stationarity issue that it attributes to the earlier formulation in [1,2] |
| **Soft-RL interpretation** | Our related work explicitly analyzes the connection between policy-based GFlowNet methods and Soft Actor-Critic (SAC) [5], and between value-based GFlowNet methods and Soft Q-learning (SQL) [4],  where SAC and SQL are the standard policy-based and value-based formulations of Soft RL, respectively. | [9] presents a Soft-RL interpretation as a central motivation and conceptual contribution of the work. | Building on the notation of [10], which is a SQL-style reformulation of the value-based Detailed Balance (DB) method, [9] presents an SAC-style formulation. Since the value function $V$, advantage function $A$, and value-learning objective are adopted from [1,2], the resulting SAC formulation is mathematically indentical to the formulation already established in [1,2]. Nevertheless, [9] presents this formulation as a contribution of the present work without properly acknowledging its prior establishment in [1,2].  |
| **Trust-region policy objective** | [1] introduces a soft formulation of original TRPO [7]. | [9] proposes an Ent-PPO surrogate objective. | Applying the standard PPO approximation techniques in [6] to the Soft-TRPO objective in [1] yields the Ent-PPO objective in [9] directly. Nevertheless, Nevertheless, [9] presents Ent-PPO as a new methodological contribution without clearly acknowledging that its objective follows directly from the Soft-TRPO formulation in [1] through the standard PPO approximation. |
| **Performance-Difference Equation** | Lemma C.2 of [1] (Lemma 5 of this ICLR version) provides a  Performance-Difference Lemma (PDL) that extend the original PDL for regular MDP (with invertiable transition matrix $P$) and fixed reward $r$ to absorbing MDP (singular $P$) and non-fixed rewards $r$ | Lemmas F.1 and F.2 of [9] are presented as performance-difference results for GFlowNets. | After aligning the corresponding notation and sign conventions, the expressions in [9] coincide with the earlier result in [1]. Therefore, [9] presents a result already established in [1] as a new contribution of the present work. |
| **Performance bound** | Theorem 3.6 of [1] (Theorem 2 of this ICLR version) derives a performance bound, based on which the Soft-TRPO objective derived. | [9] presents Proposition F.5 as the corresponding performance bound for its Ent-PPO derivation. | The proof strategy closely parallels the earlier Soft-TRPO analysis and the standard TRPO performance-bound argument. The result in [9] appears to use a looser form of the bound, making the final expression look different while retaining essentially the same underlying derivational structure. |

---
## References

[1] Niu, P., Wu, S., Fan, M., and Qian, X. (2024). GFlowNet training by policy gradients. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 38344–38380. PMLR.

[2] Niu, P., Wu, S., and Qian, X. (2026). Evaluating GFlownet from partial episodes for stable and flexible policy-based training. In The Fourteenth International Conference on Learning Representations.

[3] Tristan, D., Padideh, N., Nikolay, M., Doina, P., and Yoshua, B. Discrete probabilistic inference as control in multi-path environments. In Uncertainty in Artificial Intelligence, pages 997–1021. PMLR, 2024.

[4] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In International conference on machine learning, pp. 1352–1361.
PMLR, 2017.

[5] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861–1870. PMLR, 2018.

[6] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017b). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

[7]  Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. (2015a). Trust region policy optimization. In International Conference on Machine Learning, pages 1889–1897. PMLR.

[8] Achiam, J., Held, D., Tamar, A., and Abbeel, P. Constrained policy optimization. In International conference on machine learning, pp. 22–31. PMLR, 2017.

[9] Anna Zykova-Myzina, Timofei Gritsaev, Daniil Tiapkin, Nikita Morozov.  Proximal Policy Optimization for Amortized Discrete Sampling,   ICML 2026 Workshop on Structured Probabilistic Inference & Generative Modeling.

[10] Daniil Tiapkin, Morozov Nikita, Naumov Alexey, and Vetrov P Dmitry. Generative flow networks as entropy-regularized rl. In International Conference on Artificial Intelligence and Statistics. PMLR, 2024.

---
## 2. Comparision Details

(a) **The four observations below show that the claimed distinctions by [9], regarding the choice of reward, its effect on the resulting value  $V^\pi$ and advantage $A^\pi$, and the policy gradient computed based on the two function, is primarily notational and interpretational, rather than a substantive methodological difference from [1,2].**

1. For GFlowNets, the horizon is finite and  $\pi(s'|s)=\pi(a|s)$, and $r(s,a)=r(s,s')$  for $a=(s\rightarrow s')$ ( see the beginning of Section 2 in [9]). The authors set $\alpha=1$ and $\gamma=1$ (see the beginning of Section 4 in [9]).  The soft value and avantage functions in e.q. (1,3) of [9] are then become $$V^\pi(s):=E_\pi\left[\sum_{t=0}^{T-1}\log \pi_B(s_t|s_{t+1})-E_{s_{t+1}}[\log \pi(s_{t+1}|s_t)]\mid s_0=s\right]=E_\pi\left[\sum_{t=0}^{T-1}\log \pi_B(s_t|s_{t+1})-\log \pi(s_{t+1}|s_t))|s_0=s\right],$$
   $$A^\pi(s,a):=\log \pi_B(s|s')+E_{p(\cdot|s,a)}[V^\pi(s')]-V^\pi(s)-\log \pi(s'|s)$$
   
   where  $\log \pi_B(s|s')$ is interpreted as reward $r(s,s')$ in e.q. (8) of [9].  However,  the definitions are indentical to $-V_F$ and $-A_F$ in [1]. Here, $$V_F(s):=E_\pi\left[\sum_{t=0}^{T-1}\log\frac{\pi(s_{t+1}|s_t)}{\pi_B(s_t|s_{t+1})}\mid s_0=s \right], \text{and } A_F(s,a):=\log\frac{\pi(s'|s)}{\pi_B(s|s')}+E_{p(\cdot|s,a)}[V_F(s')]-V_F(s))$$

   where $\log\frac{\pi(s'|s)}{\pi_B(s|s')}$ is intepreted as reward $r(s,s')$. 

    * Actually, [2] does not specify a particular step-wise form of the value function. Instead, they defines the value function more broadly in terms of the trajectory-level ratio, 
       $$V^\pi(s):=E_\pi \left[\log\frac{P_B(\tau)}{P_F(\tau|s_0)}| s_0=s \right ]=E_\pi\left[\sum_{t=0}^{T-1}\log \frac{\pi_B(s_t|s_{t+1})}{\pi(s_{t+1}|s_t)} \right ]$$
      
      which can be equivalently decomposed as the summation of either of the reward forms discussed above.   
      
    * Consequently, the claimed non-stationarity distinction cannot be attributed to a different value or advantage formulation, and [9] does not establish why merely reinterpreting the same quantities through different notations removes the claimed issue.

3.  As acknowledge by [9](  see the end of Section 2 in [9]), existing value-based methods like Sub-TB allow varing $\pi_B$  during training ( i.e.  learned jointly with $\pi_F$). This means that the Soft-RL [4,5] reward $r(s,s'):=\log \pi_B(s|s')$ varies during GFlowNet training. 

    * Therefore, if $r(s,s')$ depending on the varing $\pi_F(s'|s)$ make prior policy-based methods [1,2] non-stationary,  should the same criticism also apply to all value-based method including DB, TB, and Sub-TB? In paritical, they are equivalent to Soft-Q[4] formulations with $r(s,s')=\log \pi_B(s|s')$ under the established equivalences in [3,10].

4. According to Appendices B.5 and B.6 in [1], the Vanilla Policy-Gradient (VPG) for policy-updating reduces to TB or DB updates when the bias-variance control parameter $\lambda$ is set to $1$ or $0$, respectively.  If this is the case, are the authors also claiming that the well-known stationary methods,  TB, DB and Soft-Q [4] variant of DB [10], all suffer from the same non-stationarity issue?

5. Besides,  [9] acknowledge that Sub-EB [2] objective used for value-function updating takes essentially the same form as Sub-TB (see words above e.q. (15) in [9] ). If Sub-TB is free from the claimed non-stationarity issues, then why would Sub-EB [2] suffer from such issues?


Taken together, these observations reveal an internal inconsistency in the stated motivation of [9]: the criterion used to label prior policy-based methods as non-stationary also applies to value-based formulations that [9] treats as stationary or stable.

------------------

(b) **Second, one core contribution claimed by [9] is to reinterpret policy-based GFlowNet methods from the perspective of soft RL. However, the relationship between GFlowNets and soft RL, including Soft-Q Learning  (SQL) [4] and Soft Actor-Critic (SAC) [5], has already been clarified in [2,3]**.  To be specific,

 5. Appendix A.6 of [2] shows that policy-based methods for GFlowNet correspond to SAC[5].  [2,3] both shows value-based methods, including DB, TB, and Sub-TB, and the Soft-RL variants [10] correspond to SQL[4]. Therefore, the Soft-RL perspective presented in [9] does not constitute a new conceptual framework for policy-based GFlowNets. It falls directly within the Soft-RL analysis already established in [2,3], including the correspondence to SAC and Soft-Q learning.

 6. Given equivalence between SAC and policy-based GFlowNet under per-step objectives for learning $V^\pi$ [2],  the same non-stationarity argument in [9] would also apply to the SAC. [9] does not explain why the  problem should arise in the policy-based representation of [1,2] but disappear merely by rewriting the same mathematical formulation under different notation.

------------------

(c) **Thirdly,  the three observations below show that Ent-PPO can be derived directly by applying the standard PPO approximation techniques in [6] to the Soft-TRPO formulation for GFlowNets introduced in [1].  However, the paper does not explicitly acknowledge this connection, nor does it adequately credit the theoretical foundation of Soft-TRPO[1] on which Ent-PPO is built.**

7.    The TRPO obejective for GFlowNet is[1] :
  
      $$\min_\theta  E_{\tau\sim \pi(\theta)}\left[\sum_t A_F^{old}(s_t,s_{t+1})\right], \\ \text{s.t.}  KL(\pi_\theta|\pi^{old})<\delta $$
  
      where  $A_F^{old}$ is the soft advantage function of $\pi^{old}$  and equal to $-A^{\pi_{old}}$. Following the standard PPO derivation in Section 2.2 of [6],  this contrained objective can be converted to a non-contrained objective:
      $$\max_\theta  E_{\tau\sim \pi^{old}}\left[\sum_t  \rho_t (\theta)A^{\pi_{old}}(s_t,s_{t+1})- \beta KL(\pi_\theta|\pi^{old})\right ], \quad \rho_t(\theta)=\frac{  \pi_{\theta}(s_{t+1}| s_t) }{ \pi_{old}(s_{t+1}| s_t) }
      $$

      which concide with equation (20) of [9].  Further using the Cliping tricks in Section 3 of [6],  arrive at the Ent-PPO objective in [9].

8.  Lemma F.1 (Soft performance-difference lemma) and Lemma F.2 (Surrogate as state-distribution swap) in [9], specifized to finite-horizn and deterministic-transition setting already exisit in  Lemma C.2. in [1] (Lemma 5 of this ICLR version).  However, the relationship between these derivations is not mentioned.  Lemma F.2, presented as a reformulation of Lemma F.1, states that:
   
     $$J(\pi_\theta)-J(\pi_{old})=\sum_{t=0}^{T-1} E_{s_t\sim d_t^{\pi_\theta}}[\phi(\pi_\theta,s_t)],   \quad \phi(\pi_\theta,s)=E_{s'\sim \pi_\theta(\cdot|s)}[A^{\pi_{old}}(s,s')]-KL(\pi_\theta(\cdot|s)\|\pi_{old}(\cdot|s)),$$

     Using the same notation, Lemma C.2. in [1] states that  if   $KL(\pi_\theta(\cdot|s)\|\pi_{old}(\cdot|s))<\zeta $:

     $$J(\pi_{old})-J(\pi_\theta)\leq T\cdot E_{s\sim d^{\pi_\theta},s'\sim \pi_\theta(\cdot|s)}[-A^{\pi_{old}}(s,s')]]+T \zeta.$$
     Otherwise,  $$J(\pi_{old})-J(\pi_\theta)=T\cdot E_{s\sim d^{\pi_\theta}, s'\sim \pi_\theta(\cdot|s)}[-A^{\pi_{old}}(s,s')]]+T\cdot E_{s\sim d^{\pi_\theta}}[KL(\pi_\theta(\cdot|s)\|\pi_{old}(\cdot|s))].$$

     Since $d^{\pi_\theta}:=\frac{1}{T}\sum_t d_t^{\pi_\theta}$, the two expressions coincide after aligning the sign convention. Presenting these lemmas as part of the new theoretical development of [9] without identifying their direct predecessor in [1] creates a misleading impression of theoretical novelty.


10. The proof of Proposition F.5 in [9] closely parallels that of Theorem 3.6 in [1]. The main difference is that Proposition F.5 employs a looser bound, which makes the final expression look different while retaining essentially the same underlying derivational structure. This does not appear to constitute a substantive theoretical distinction, particularly because Theorem 3.6 in [1] provides a tighter bound, which taking a similar form to the standard TRPO performance bounds[8].

    There is also a error in the proof of Proposition F.5. [9] cites Lemma 3 of [7] as the “standard occupancy-mismatch inequality for finite-horizon MDPs,” but Lemma 3 of [7] does not state this result. The result actually required by this proof appears explicitly as Lemma B.5 of [1]. This makes the lack of attribution particularly consequential: the proof not only closely follows the Soft-TRPO derivation of [1], but also relies on a inequality explicitly established in [1] while citing a lemma that don't state the required result.

**All comparisons above are based solely on publicly available versions of the respective works.**

