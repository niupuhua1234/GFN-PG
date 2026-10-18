# Technical Comparison and Originality/Attribution Concern

This document preserves a public technical comparison concerning the relationship between our prior works, *GFlowNet Training by Policy Gradients* (GFN-PG) and *GFlowNet Policy Evaluation via Subtrajectory Balance* (Sub-EB), and the subsequent work *Proximal Policy Optimization for Amortized Discrete Sampling*.

The comparison below is based solely on publicly available versions of the respective works. The original public record is available on OpenReview:

**[Original OpenReview technical comparison](https://openreview.net/forum?id=k2lkeCCfRK&noteId=R9ob5DJqcf)**

This GitHub document serves as a persistent mirror of that technical record. Substantive updates can be tracked through the Git commit history.

---

## 1. Public chronology

Our work *GFlowNet Training by Policy Gradients* was submitted to ICLR 2024 on **September 23, 2023**, made publicly available on OpenReview on **October 13, 2023**, and subsequently accepted at **ICML 2024**.

Our later related work, *GFlowNet Policy Evaluation via Subtrajectory Balance* (Sub-EB), appeared at **ICLR 2026**.

The later paper *Proximal Policy Optimization for Amortized Discrete Sampling* was made public after these works.

This chronology is relevant because several conceptual, methodological, and theoretical components presented in the later paper have direct or closely related counterparts in the earlier works.

---

## 2. Summary of the concern

The overlap goes substantially beyond a general similarity in topic or an incomplete related-work discussion.

In particular, the later work contains direct or closely related counterparts to earlier results on:

- the value and advantage formulation for policy-based GFlowNets;
- the Soft-RL interpretation of GFlowNet learning;
- the role of policy-dependent log-ratio rewards and the resulting non-stationarity discussion;
- the relationship between policy-gradient objectives and standard GFlowNet objectives;
- the relation between Sub-EB and Sub-TB style value learning;
- the Soft-TRPO formulation for GFlowNets and its standard PPO approximation;
- the finite-horizon performance-difference result; and
- the corresponding performance-bound derivation.

Most importantly, the central Ent-PPO objective in *Proximal Policy Optimization for Amortized Discrete Sampling* can be obtained directly by applying the standard PPO approximation to the Soft-TRPO objective introduced in *GFlowNet Training by Policy Gradients*, after aligning the sign and notation conventions.

Based on the publicly documented equation- and theorem-level correspondences below, **we consider the presentation of previously established conceptual, methodological, and theoretical results as new contributions in the later work to constitute serious academic misconduct concerning originality and attribution, rather than merely a matter of missing citations or incomplete related-work discussion.**

---

## 3. Value and advantage formulations

The later paper defines a Soft-RL value function \(V^\pi\) and advantage function \(A^\pi\) for GFlowNet training.

Our earlier GFN-PG work defines the corresponding forward value and advantage functions \(V_F\) and \(A_F\). After aligning the notation and sign convention used by the two papers, the quantities satisfy

\[
V^\pi = -V_F,
\qquad
A^\pi = -A_F.
\]

Thus, in the finite-horizon GFlowNet setting considered by both works, the difference is primarily one of notation and interpretation rather than a substantively new value/advantage construction.

The later Sub-EB work also gives a trajectory-level ratio formulation that decomposes into the same type of per-step log-ratio quantities used in the Soft-RL representation.

---

## 4. Policy-dependent rewards and the non-stationarity argument

A central motivation in *Proximal Policy Optimization for Amortized Discrete Sampling* is that the policy-based GFlowNet formulation involves a policy-dependent reward of the form

\[
r(s,s') = \log \pi_B(s\mid s') - \log \pi_F(s'\mid s),
\]

or the equivalent sign convention, and that this creates a non-stationarity issue.

However, existing GFlowNet methods already allow the backward policy \(\pi_B\) to change during training. In value-based formulations such as Sub-TB, changing \(\pi_B\) also changes the corresponding log-ratio reward.

Therefore, if policy dependence of the per-step log-ratio is taken as sufficient to characterize the policy-based formulation as non-stationary, the same criterion would also have to be applied consistently to value-based GFlowNet formulations in which \(\pi_B\) is learned or updated.

The later paper does not clearly explain why the claimed difficulty should be specific to the policy-based representation while analogous policy dependence is already present in established value-based GFlowNet formulations.

---

## 5. Vanilla policy gradient reduces to established GFlowNet objectives

In our earlier GFN-PG work, the vanilla policy-gradient update was shown to reduce to established GFlowNet training objectives under particular choices of the trace/credit-assignment parameter.

In particular, the appendices show that the policy-gradient update reduces to:

- a Trajectory Balance (TB)-type update at one endpoint; and
- a Detailed Balance (DB)-type update at the other endpoint.

This result directly connects policy-gradient GFlowNet training to previously established balance objectives.

Consequently, a sharp conceptual distinction between a supposedly non-stationary policy-gradient formulation and stationary TB/DB-style formulations is difficult to maintain without reconciling these equivalences.

---

## 6. Sub-EB and Sub-TB

Our Sub-EB work introduces a value-learning objective based on subtrajectory evaluation.

Its objective has essentially the same structural form as Sub-TB, while providing a policy-evaluation interpretation and enabling learning from partial episodes.

This creates another inconsistency in the later paper's non-stationarity motivation: if Sub-TB is regarded as free of the claimed problem, it is unclear why the closely related Sub-EB formulation should inherit a fundamentally different non-stationarity issue solely because of its policy-evaluation interpretation.

---

## 7. Soft-RL interpretation

Our prior work explicitly connects policy-based GFlowNet learning to Soft Actor-Critic (SAC)-style policy evaluation and optimization, while relating value-based GFlowNet methods to Soft Q-learning.

Accordingly, the Soft-RL interpretation in the later paper has direct precedent.

Returning to the motivation of the later paper, given the established equivalence between SAC and policy-based GFlowNet methods under the corresponding per-step objectives for learning \(V^\pi\), the same non-stationarity argument would necessarily also apply to the SAC formulation.

The later paper does not explain why the claimed problem should arise in the policy-based GFlowNet representation but disappear after rewriting the same mathematical formulation in the equivalent SAC representation.

---

## 8. Ent-PPO follows directly from the earlier Soft-TRPO formulation

The most direct methodological relationship concerns the later paper's central Ent-PPO objective.

Our earlier GFN-PG work introduces a Soft-TRPO formulation for GFlowNet training. In simplified notation, the update has the form

\[
\min_{\theta}
\;
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_t
A_F^{\mathrm{old}}(s_t,s_{t+1})
\right]
\]

subject to a trust-region constraint of the form

\[
D_{\mathrm{KL}}
\!\left(
\pi_\theta
\;\|\;
\pi_{\mathrm{old}}
\right)
< \delta .
\]

With the sign correspondence

\[
A_F^{\mathrm{old}}
=
- A^{\pi_{\mathrm{old}}},
\]

this is the same policy-optimization structure to which the standard PPO approximation is normally applied.

Introducing the usual importance ratio

\[
\rho_t(\theta)
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\mathrm{old}}(a_t\mid s_t)
},
\]

and replacing the trust-region update with the standard PPO clipped surrogate yields the familiar form

\[
\mathbb{E}
\left[
\min
\left(
\rho_t(\theta) A_t,
\;
\operatorname{clip}
\bigl(
\rho_t(\theta),
1-\epsilon,
1+\epsilon
\bigr) A_t
\right)
\right],
\]

with the corresponding Soft-RL/GFlowNet advantage.

Thus, **the Ent-PPO objective can be obtained directly by applying the standard PPO approximation to the earlier Soft-TRPO objective**.

In our view, presenting this resulting objective as an independent methodological contribution without clearly identifying its derivational dependence on the earlier Soft-TRPO formulation materially misrepresents the provenance of the method.

---

## 9. Performance-difference result

The later paper presents finite-horizon performance-difference lemmas.

Our earlier work already established the corresponding finite-horizon performance-difference relationship.

Using the occupancy measure

\[
d^{\pi_\theta}
=
\frac{1}{T}
\sum_{t=0}^{T-1}
d_t^{\pi_\theta},
\]

and aligning the sign conventions for the advantage functions, the expressions coincide with the earlier result in the finite-horizon setting.

Accordingly, these later lemmas reproduce a result already established in the prior work rather than constituting a new theoretical contribution.

Presenting such results as new theory without clearly identifying the earlier counterpart creates a misleading impression of novelty.

---

## 10. Performance-bound derivation

The later paper also presents a performance bound whose derivational structure closely parallels the performance bound established in our earlier work.

The main apparent difference is that the later result uses a looser bounding step, while following the same overall finite-horizon occupancy-mismatch and advantage-based argument.

The later proof also attributes a finite-horizon occupancy-mismatch inequality to a standard TRPO lemma. However, the corresponding finite-horizon inequality is explicitly developed in the earlier GFN-PG analysis.

This further strengthens the concern that the theoretical development in the later paper closely follows previously established results without adequately distinguishing the earlier derivation from the genuinely new content, if any.

---

## 11. Overall assessment

Taken together, the issues above go substantially beyond ordinary overlap in topic.

The value and advantage formulation, the Soft-RL interpretation, the policy-optimization objective, the performance-difference result, and the performance-bound derivation presented in *Proximal Policy Optimization for Amortized Discrete Sampling* all have direct or closely related counterparts in our prior works.

Most importantly, the central Ent-PPO objective follows directly from applying the standard PPO approximation to the Soft-TRPO objective introduced in *GFlowNet Training by Policy Gradients*.

Nevertheless, the later work presents several of these elements as its own conceptual, methodological, and theoretical contributions without clearly distinguishing the previously established results from its genuinely new contributions.

For these reasons, **we consider this a serious originality and attribution problem and, based on the publicly documented technical evidence, serious academic misconduct concerning originality and attribution rather than merely a citation or related-work issue.**

---

## 12. Public record

The detailed comparison was originally posted publicly on our ICLR OpenReview page:

**[OpenReview technical comparison](https://openreview.net/forum?id=k2lkeCCfRK&noteId=R9ob5DJqcf)**

This GitHub document is intended to preserve the same technical record in a repository controlled by the authors of the earlier work. Any substantive revision to this document is visible through the public Git commit history.

All comparisons in this document are based solely on publicly available versions of the respective works.

---

## References

**[1]** Puhua Niu, Shili Wu, Mingzhou Fan, and Xiaoning Qian.  
*GFlowNet Training by Policy Gradients.*  
International Conference on Machine Learning (ICML), 2024.  
https://proceedings.mlr.press/v235/niu24c.html

**[2]** Puhua Niu et al.  
*GFlowNet Policy Evaluation via Subtrajectory Balance.*  
ICLR 2026.

**[3]** Anna Zykova-Myzina, Timofei Gritsev, Daniil Tiapkin, and Nikita Morozov.  
*Proximal Policy Optimization for Amortized Discrete Sampling.*  
SPIGM @ ICML 2026.  
https://openreview.net/forum?id=ODbzTJmgp3

**[4]** John Schulman et al.  
*Proximal Policy Optimization Algorithms.*  
arXiv:1707.06347.

**[5]** John Schulman et al.  
*Trust Region Policy Optimization.*  
International Conference on Machine Learning (ICML), 2015.
