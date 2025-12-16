Okay, so, welcome to the…
It's not really a lecture, I don't want to call it a lecture for reasons I explained in my chat. The lecture is also shortened
for the same kind of reason, so what I'm planning to do today is to finish what I was discussing last week about reinforcement learning, try to cover a…
At least at a high level, the schemes associated with reinforcement learning fine-tuning, and also give you some kind of guidance with respect to your final exam.
I think I just posted the solution… no, I just posted the solutions that I gave you, of the problems I gave you last week.
And so, and also, I have some,
suggestion with respect to the scope of the final exam, which I would like to make a kind of a slight modification there, okay? Which probably is going to be better for all of you, I think, that modification. Are you able to see me? No, you cannot… you cannot see me.
Yes.
No? Okay, alright, so here we go.
Alright, so, I kind of wanted to…
So for the… because, you know, we are not really in person, I cannot really understand, you know, if what I'm going to be discussing is understood to all of you. So, I will, you know, I will… basically, if you have any questions, please don't hesitate to interrupt.
I'm not able to see the chat.
And I'm not able to see either the Discord or the chat in Zoom, okay, just to let you know.
Okay, so…
I think last week, I'm not sure where we started, I think we started further down, if you like, but,
we went through the MDP discussion already. We understood, you know, what is the return.
which is kind of essential to understand this expectation of the objective function, for the… and the associated, if you like, estimation or prediction problem of the state value function. And,
I, what I, was trying to do later, I think this is basically where we started.
We started with some kind of attempt to estimate it using a very simple kind of environment. As I told you, the simple environments kind of help us to understand what's really happening.
And be able to…
implement algorithms, let's say, from scratch in Python, and on the other extreme is really an LLM, which is going to be a much more difficult experimentation kind of scheme, because you do not really fully get to appreciate all the state
Of the… of the environment, and… and the experimentation becomes a bit more challenging. In the middle ground.
If you don't do LLM, but you do small language models, I think there's still… I think it's a useful kind of middle ground, and Hugging Face has published a variety of
Small language models that you can use as kind of.
small baselines to experiment with these techniques that we are discussing here. Okay, so we understood that the estimation problem of the value of states in any environment, at the end of the day is a linear equation.
Given by the Bellman value function equation. And, in an attempt to solve it.
We understood that it is basically a system of linear equations, and we also solved it in an iterative way, which I suggested for you to always kind of migrate to every time you're facing an estimation kind of problem.
Then we looked at the state action value function.
Which is, the… not only represents the value of a state, but also represents the value of taking a specific… and committing a specific action from that state, and following, after that policy pi.
And again, the famous kind of expectation in front of us, because of the very fact that we have, you know, after state S and taking action A, we are facing a random process.
Well, every single time instant in this kind of random process is going to be a sort of associated with a reward, and therefore defines, if you like, a return.
From that, from that moment until the…
kind of process terminates, or if it continues to infinity, then obviously the return is going to be
Some kind of a geometric kind of series of, discounted rewards.
Okay, so that is basically the… the equation, and we kind of understood, again, the same…
backup tree, that we have seen earlier, but instead of now backing up from the values of next states to the value of the state S,
We are backing up from the Q pi of S prime comma a prime to the corresponding Q function value of the Q function in the state S and A.
And so the, this is basically what we call bootstrapping.
Where,
we effectively bootstrap the value of a state and action, in this case, from values that are in other states and other actions, taking on those states another different action, potentially, A'.
Whatever the S prime and A prime is.
So this was, effectively, the…
sort of dynamic programming type of equations that will actually allow us to estimate the sort of values, the V function and the Q function. And then we migrated to control, because ultimately we are trying to
solve the problem, and solving the MDP problem involves coming up with a pi star.
And, coming up with a pi star involved, now nonlinear equations.
Because effectively we said fairly intuitively that, the value
of a state, if you follow from that state onwards a maximum kind of the best policy, the optimal kind of policy, is, the…
maximum possible value of the Q values that you see around you, following that kind of same optimal kind of policy. And therefore, we wrote, if you like, the first equation, and we sort of started
writing down the Bellman, optimality equations now, that, because of the max.
Operator over there are becoming nonlinear.
And, we,
I kind of wanted to solve, to provide an example, and we went to the recycling robot example in class, and wrote down exactly for this specific example what will be the V-star of high and the V-star of low, which we didn't write, but I asked you to do it at home.
Make sure that you understand how to do it. Okay, so… so two systems of nonlinear equations that can only be solved with a technique called policy iteration. I didn't write anything on the policy iteration here on this whiteboard, but I went through it,
explaining, if you like, the contraction in your, in your class, site. So I went through, the policy iteration.
over here, and I brought up this, if you like, this diagram, and I discussed how the policy duration is implemented using a simple grid world problem.
That, someone may give you, and ask you to, come up with, sort of the optimal value function after, I don't know, two, three iterations. I have no idea. But,
This is basically the… the part of the site that I asked you to,
To understand a little bit, to, to understand how, value, function, acting on the value function in a greedy way, can actually eliminate some of the actions of the, in specific kind of states, and
Therefore, result into a policy at the end, which is called optimal policy, because despite the fact that the value function may actually change from one iteration to another after a while, the relative,
benefits or relative values between adjacent states and things like that that drive the decision-making process of the agent, do not, and therefore the optimal policy may have converged much sooner than the value function will.
But nevertheless, or stay the same, effectively.
Okay, so this was basically the way to solve the MDP, was just policy iteration. We didn't do anything on the value iteration, which is yet another scheme over here. And, we can,
We then moved into reinforcement learning, okay? So, reinforcement learning, we studied two major methods in reinforcement learning. The first one is the Monte Carlo method.
And the other is, the temporal difference method. And in reinforcement learning, our ability to Expun.
Let me just go to reinforcement Learning.
Let's go to reinforcement learning. So.
What we… what we were, effectively, up to this moment in time, or sort of, studying, is, if you divide, if you like, the… all the problem space into this kind of, shape, where one axis is the…
sort of, breadth, of how much, in terms of number of states you are able to expand, in order for you to be able to understand what you could potentially go.
definitely the dynamic programming is, as I told you, a bread-first type of expansion kind of… sort of… of this kind of backup tree, because you know exactly all the states it could potentially migrate to, and you know exactly all the rewards that the environment will actually give you.
And therefore, you can use this kind of information to actually do a full backup sort of,
estimation of the value of a state, given all the subsequent kind of states. So this is basically the top right
sort of corner over there, this kind of diagram, and we do shallow backups in a sense that we are doing one problem at a time, one sub-problem at a time. Effectively, that's the definition of dynamic kind of programming, all the way to the point where we have brought
the values, calculated the values in this kind of iterative way, as I mentioned, across the whole environment of all the states that the environment can migrate to.
So the Monte Carlo method
is on the bottom right corner, because in Monte Carlo, we have absolutely no idea
what is the transition model, which states we can migrate to, and what reward do we expect out of this kind of… each of these kind of states. Therefore, the only thing we can do is to treat the environment as a black box.
Treating the environment as a black box means that we are just basically taking action and let the environment give us the new state and the reward, and then from that new state, take another action, let the environment give us whatever it wants.
And therefore, at some point, we may terminate. That's basically the shape of this kind of backup tree. It's like a vertical line over here, and I drew it a little bit differently because I was accounting for the fact that
You may take other actions in the process, but,
Here, for one action, it is basically this,
this diagram here. So we are effectively generating trajectories.
So, the state S may participate, or S double prime, as I can see it here, and S double prime, as I see it here, can participate in multiple trajectories, and therefore, the only thing we can do, as I explained, is to calculate the sample mean.
Of the returns from each of the states that we are visiting, and of course, we need to count also, to calculate the sample mean, the number of times where we are visiting these kind of states.
And, the, incremental sample mean approximation, we, as an equation.
is, already has been discussed in the sort of Kalman filter kind of lecture, and resulted in this kind of equation that we have borrowed and wrote the incremental sample mean update of the value function here in a very… in an identical way.
So, remember that, the…
GT here, which is shown in the purple box, is the sample return for this specific trajectory. In other words, what kind of returns I'm actually getting get, starting from time t, and, the, until I terminate. And, as the…
V pi of S is the… V pi of S here is the previous sample mean in this kind of a difference.
So this is… these are the… I hope you recognize the similarities of these two equations, right?
Okay. Alright, so, that was basically the…
if you like the Monte Carlo method, and, and the temporal difference method, and I, I went, in, your, book, site, on the, on the course site, and I actually showed you some,
A problem, sorry, it's a problem.
Over here, it's a very simple kind of problem, and someone may also ask you certain questions in the final exam about
as to, okay, what is the Markov chain… sorry, Monte Carlo is doing over here, right? So, obviously, we are generating thousands of trajectories, and depending on the value of A, or alpha, which is the forgetting factor over here, we may actually get different
curves of convergence. One thing is, certain.
The fact that the fact that we are generating all this kind of a deep.
sort of trajectories all the way until we terminate. One disadvantage of this kind of scheme that actually led us to the definition of a temporal difference is the fact that we have to terminate in order to calculate the return.
Okay, so the sample return. Otherwise, we do not know the return. So the one approach that we have taken, going into the TD method, is to say, okay.
Is there any way, that we can, sort of bootstrap
that… what we call the TD target over here, and this bootstrapping is happening from the fact that whatever state I will end up, that state must have a value associated with it from previous experiences.
And this value, whatever it is, I'll pick it up and I'll plug it in over here, and I will effectively do a one-step bootstrap of the value of the S of T using that one, and of course, I will increment
The sample mean, using, this So, the TD target effectively
Takes the place of the new piece of information that we are getting.
from that kind of a bootstrapping, single-step bootstrapping here. The single step is coming because of this zero over here, which is a special case of what we call later a TD of lambda.
And in the TD of lambda, in order to sort of understand the TD of lambda, is… we need to understand first the TD of the n steps, the multiple steps that, are… effectively can be used to bootstrap that
value of all the states that we have visited. So, if we do not bootstrap in the first step, and we bootstrap after two steps, then this is a two-step kind of TD target, a three-step TD target, all the way to n-step TD target.
And then we can actually write down this kind of equation.
Where…
The, update of the sample mean, happens over, in a, in a kind of a different time scales compared to the…
sort of returns that we are able to compute. And, so we kind of decouple the time that the actions are taking, and of course, and value function updates are done, because obviously we're doing these V-function estimates, because at the end of the day, we need to find optimal ways to act. That's the… that's the reason.
Okay, so the TD of lambda now is,
is a kind of a weighting of all this kind of n-step returns that we can actually produce, so I'm just going to write down the equation. It's not an equation which is,
sort of,
it's kind of a complicated equation, so I'm just going to continue. I'm not going to call it here a different lecture. I'm sure we'll continue and finish this discussion. So, the GD of lambda as an equation
And, is, equal to 1 minus lambda.
summation from n is equal to 1 to infinity.
Lambda to n minus 1.
GT of n. Well, GT of n, we discussed last time what it is, and you actually see the equation over there.
Right. So, when Lambda… Is equal to zero.
when lambda is equal to 0, we are effectively going back to the TD of 0.
And,
when the lambda is equal to 1, we are going to… back to Monte Carlo. So the TD of lambda
with this specific weighting of the returns that we get out of the end step kind of returns, is able to give us the two extremes. And for any lambda value which is between 0 and 1,
We will get,
we will get something in between. Okay, so the equation that it is something in between is kind of a self-evident.
But I want to stay a little bit,
I want to stay a little bit,
on this equation, let's see, V… D.
plus N.
of ST is equal to VT plus n minus 1 of st.
plus alpha.
GT of lambda.
minus V, T plus N.
minus 1 off SD.
Okay, so this is the… this is the TD of lambda.
Kind of update of the value.
And as I said, we…
it goes with this kind of weighting, which is an exponential kind of weighting, and it looks like this, if I have the diagram here.
The diagram… Lugust… Capital.
the diagram.
Looks like this.
Okay, so… The diagram is… over T.
And, this is… The 1 minus lambda.
And the weights… of the returns.
Cool.
Obviously, the weights are going to be at some point, if I terminate, it's going to be truncated, right?
But, this is basically the…
the returns that we will get, the weight of the returns that we get over multiple kind of steps. So, as you can see, the weights are adjusted depending on how many steps we have,
We do the estimates.
As you can see also from this kind of equation.
So I bypassed a little bit of the discussion, because, one, anything I discuss today, anything, that I discuss, effectively.
In fact, it's… let me call this line. Everything below this line is out of scope, anything above this line is in scope for the final.
So, whatever we discuss in person, it's in scope. Anything, today is out of scope.
Yeah.
Okay, so,
So we kind of go into the direction where we are trying to understand, you know, in terms of algorithmic kind of content. Okay, we have now the base, the basic kind of algorithms, which are called value functions. I want to partition the space of reinforcement learning now.
So if I, if I kind of partition the algorithmic space in reinforcement learning.
I will, probably, and I think I have… Wow.
Okay, hold on a second, because I…
Lost my notes here in this… Okay, here they are.
So I wanted to, sort of partitioned the reinforcement learning into three parts. The first part
Is the so-called, Value.
based algorithms.
The second, part is, the second part is… policy-based algorithms.
And the third part is… model.
Based.
algorithms.
Okay.
And I think there is a…
a nice diagram, over here in the book that I recommended you to, which is a bit more Python-oriented, to study, kind of, reinforcement learning.
Let me just,
Okay, yeah. So let me, let me write down a little bit this. So, these are all categories that come under
The headline of, model-free methods.
These are model-free methods.
And obviously, this is the model-based methods, okay, to state the obvious.
So now… Out of these kind of three boxes, so just examples of this,
of these examples of this kind of algorithms, I'm going to write it here on the side, is,
Oops.
is this…
Sarsa?
I have some notes on the side, but because PPO is not really based on this, I will not cover it. So SARSA is a method that will allow us to do the control problem in reinforcement learning. Up to this moment in time, we were dealing with estimation problems, or prediction problems.
in enforcement learning. From control, we are going to do
only control about policy-based algorithms, which most LLMs are
driven by. Okay, so we have, Sarsa, we have, Deep, Deep.
Q networks,
And so on. We have lots of categories over there in value-based kind of algorithms. In policy-based algorithms, we have one fundamental algorithm that at least we need to… I need to do one, and this one is called reinforce.
Which, of course, is a policy gradient kind of method. And, so we need to look at it in detail to understand what's really happening in a policy-based algorithm, because… because…
The policy-based algorithm and the
I will call it,
The value-based algorithms are coming together. Let me…
Let me write the new force somewhere else, because I need the space here.
So, this guy is… Reinforce.
So the value-based algorithms and the policy-based algorithms are coming together to define
What we call hybrid kind of alums, or combined methods.
And, combined between value and policy standards, both of them are coming into this. So, a famous algorithm that, is,
actor, critic.
And, the, the type of methods, TRPO,
Which is a Trust Region Polish optimization.
And, PPO.
The proximal policy optimization, and we will see some elements of that a little bit later.
And, so, most of the reinforcement learning kind of fine-tuning discussion is obviously, in, on this side. Obviously, GRPO,
Was, defined, knowing all of this above.
And, greatly simplified the PPO algorithm, to… for the benefit of, efficiency.
And this is the PPO, was adopted by OpenAI.
And GRPO was introduced by DeepSeq.
Okay, so the…
So why I'm actually mentioning all that, we need to, we need to cover the gap, because if we don't cover this.
You know, we are…
we cannot really understand anything that follows, because it's a combined kind of method, okay? And I should warn you that policy-based algorithms are
Not necessarily difficult, but when you are doing about combination,
looking at combinations of this value and policy, then the situation becomes, especially on PPO, a little bit complicated. Okay, so…
All right, so let's, let's start. Any questions up to this moment in time on…
on what… where we are going, and any questions on the previous discussion about the TD of lambda?
Okay.
Alright.
So, so let's look, let's look at, Policy, what it's called?
Okay, so let's look at the… Reinforce.
And, policy gradient base.
Let me see… Gradient.
Based… algorithms.
Okay So… So I have some, write-up that I posted yesterday.
on a policy gradient kind of algorithm. This is out of scope, and,
What we're actually trying to do here is, as you probably understood, is we're trying to sort of change the policy directly.
By changing the parameters of
a network, a neural network, that it will, at its output, will produce this policy, okay? So, it will produce the policy, in other words, it will produce a probability distribution, and okay, thank you very much, all the networks we have done up to this moment in time produce a probability distribution at their output.
Changing the parameters of the network obviously changes the policy, therefore changing the probability distribution. So, if we are to,
If we are to…
Hold on a second. If we are to understand what we are trying to do here at the high level, is, for, a policy pi theta.
Let's say 1, we have a probability distribution.
Obviously, in an LLM, this probability distribution is gigantic, it's like the cardinality
Of the vocabulary set, because, the…
sort of probability distribution we have to feed into the sampler is the posterior probability of all the tokens in my vocabulary, and someone will need to sample one, that kind of satisfies the sampling kind of policy.
a sampling, I don't want to call it policy, but a sampling kind of algorithm. And so, in this case, let's assume that,
This was, A kind of an action, the equivalent of an action.
Is the sampling.
Of a new token.
Let's say, okay? So,
So what we're trying to do here with a policy kind of gradient is to say, okay, fine, we've seen this kind of action, and we have the means of evaluating, if you like, the policy, and perhaps we can actually have an algorithm that it will
do this. It will modulate
If this action did not result into receiving, let's say, a good reward, an average expected reward, let's say, moving forward, or an average kind of return from that kind of state that we are in, then we would like to
reduce the probability of this specific action, and increase the probability of another action, let's say this one, for example, and so the posterior probability distribution will change us to something like this, okay?
Okay, so next time, we are going to motivate, the, with the policy, with the, with the parameters of the… the new parameters of the network, theta2,
to produce the pi theta2 policy at the output. And so, by changing the thetas, we are going to be changing the policy. But we cannot really change the policy dramatically. As you may understand. We are trying to also to modulate how much of this kind of policy will change, so
Similar to what we have done earlier about stochastic gradient kind of methods, we are going to be changing in small increments.
Okay, does it, does it, does it,
Does it correlate with your kind of high-level understanding of this? That's basically what we're trying to do here.
So, if you write it, from FITA1 will go to… Let's say theta2.
Then pi theta 1 will go to pi theta2.
So we need to have an objective, and this objective is going to be called here J.
And, if we are to optimize this objective, I'm going to call it J star of pi of theta. I do not want to call it a loss, because I want to present the problem first from a point of view of
an optimization, which is going to be maximized, so therefore I started calling, kind of, objective.
And, later, I will convert it into a loss, because a loss is going to be, you know, obviously the negative of the objective function we will come up with. So this is basically max.
Over theta.
J of pi theta.
So this objective function will actually have the policy. It's a function of the policy, and what we will are trying to do here is to adopt the
Following update.
Which is based, the policy gradient I based…
Algorithms in general, and it will be the well-known gradient ascent.
So this… this is basically the way that to update, if you like, the…
The theta, and therefore the policy. And this is obviously our learning rate.
that… We will, optimize later.
Okay.
So, this is the… this is basically the algorithm. So the question becomes now, What is really the,
What should be really the objective?
For that specific,
algorithms to work, and we have a problem that I would like to sort of explain a little bit, and this problem is the following. In this kind of optimization kind of scheme. We need this. We need to calculate this guy, right?
And, if you see this, this thing, in all previous kind of discussions, and even in stochastic gradient descent, we were always optimizing… optimizing expectations, because
At the end of the day, in, let's say, reinforcement learning, it's a kind of a random process of returns all the way until we terminate, let's say. Either the sample mean returns or the TD of lambda returns is one and the same thing. So we will be dealing with some form of expectation. So we have a gradient of an expectation, and
The gradient of expectation is definitely
a quantity that will… what they have done is they have tried to migrate it to a simpler computable quantity using a trick called log derivative trick. So, if you see the equation.
in the textbook, the equation of updating, if you like, the theta is not looking like this, but it looks like a different kind of equation. So I'm going to explain the migration from this
gradient of J of pi of theta to the gradient of another sort of expression over here, which is going to be proven to be equivalent.
Audio shared by Okay, so at least… Yes, go ahead.
Ojas Gramopadhye: Yes, sir.
Audio shared by Could you explain where this, differentiation comes from the formula? Like, the differential of star of a pii theta.
Oh, this is J, right? J of pi of theta, right? So this, J…
Audio shared by I see, okay.
Yeah, it's J, the objective kind of function, and this j is nothing else as the expected returns that we get, as we discussed a little bit earlier, right? So this is basically… you will see
in there an expectation of the returns at G of T, right?
given the fact that you are in some kind of state. So, as the environment kind of migrates you from one state to another state and so on, you always have a sample
sample returns, that you are able to, calculate, because you're storing, if you like, the data. And therefore, average over many of these returns that you have experienced starting from that kind of state, that is basically what comes out of this, inside this, expectation, sorry, in this, in this objective.
Does it, no. Yeah, okay.
All right, so, so let's see, I have the derivation here.
In fact, I will tell you where the derivation is.
In the website as well.
So let's look at the website. I don't want to write 7 equations here and kind of bore you with this.
So this is basically the…
We're trying to get a gradient of an expectation to be graded as an expectation of gradients of a function.
That, that function is a log Function that includes their policy.
And
And so, as a probability of taking this action, and so this is basically the expectation you see above the… above the,
blue box.
Okay, so we have here a link to… the…
full derivation of this thing, but I included some smaller number of steps, but if you want to see the full
derivation, you need to go to the 231 section of the Python Reinforcement Learning Book, as I call it, okay? You understand the Python Reinforcement Learning Book is that Foundations of Deep Reinforcement Learning Theory and Practice in Python by the alphabet guys, okay?
Alright, so, effectively, make it a little bit more concrete.
We have, J.
theta.
They have a theta over there, but I skipped it because I have the theta in my policy, pi of theta.
Is equal to the, so the gradient of this Shorty.
The gradient of this.
J… 2Js.
the gradient over theta, of J of pi of theta is the gradient Over theta of the expectation.
of, Over, over all trajectories tau.
That is distributed according to policy pi of theta.
That can… you can form.
And then this is, over here you have the G.
D.
That you get out of this…
out of its trajectory, okay? So, I think that's, kind of a bit more clear now, what this thing is, okay?
So we try to find a way to write this gradient of an expectation,
As an expectation of the gradient, and the reason why we would like to do that is that,
is that the whole kind of algorithm actually allows us to do the differentiation. While before we were not able to do the differentiation, all the details are a little bit in that kind of section, so let me write section 231.
of, RL Python book.
This is where you will see the, kind of, the whole discussion there. So, but I would like to, sort of go over the final kind of formula here, which is this. So if you have, let's say, a function f , and we'd like to.
calculate, if you like, the gradient of an expected expectation of this kind of function, which is… we basically start with a definition of expectation, and we replace the expectation with an integral, which is F of X times P .
Okay, and that's… step number one, I think, is clear. Step number two is we take the gradient inside the integral.
And, then we…
Write it as a… you have a gradient of a product, a composite, if you like, function, which is the first,
fun,
first function times the gradient of the second plus the second times the gradient of the first. So this is step number three. And then, we are,
sort of, assuming that f of x does not have anything to do with the theta.
then the first one is… results into 0, so therefore, so the second one results into 0, and the first and remains now with the first expression, the first component of step number three, and then this was step number four, and then finally, we multiply and divide by P .
And, we give to step number 5, and then, we actually
write using the log derivative identity that you see over here, because, again.
We have, the gradient, let's say, of,
of, the log is 1 over
of a log of a function, P of X, is 1 over P of X times the gradient of P , which means that the gradient of the log of P of X can be written down. So we had, in step number 5, we had the ratio, right? And therefore, you can actually write it as the gradient of the log of the P , okay?
So this was basically the log derivative identity that we used to come up with the final expression, which is actually shown here as equation 7. And so, after this kind of 7, if you like, manipulations.
we are able to write the, sort of a final expression of the, creating of a j of pi of theta.
Pi of theta, as an expectation, Over all trajectories.
off the… GT?
Gradient over theta.
Well, I'm not able to write today. Gradient over theta log… by theta.
So this is basically what you will see in, as a final expression for the reinforced algorithm, or the baseline policy gradient kind of algorithm.
And the algorithm is, pretty straightforward to implement, and I have included here a kind of a pseudocode
For,
for an approach of estimating of the… adjusting, if you like, the theta based on Monte Carlo estimates across a number of episodes, and, effectively, what we are implementing here is the equation that we
we see, where the expectation is replaced here with the summation over all gradients. Okay, so the line number 8 implements the equation we just wrote.
Okay.
So we start with a…
initial alerting rate. We initialize the learning rate, we initialize the policy by parameters data. We have a neural network over here that we are trying to adjust the parameters of.
And we are… To generate new policies, and then we iterate over thousands of episodes.
Generating trajectories, using the specific policy, then using the gradient kind of approach, and executing the stochastic gradient ascent.
We are going to be changing the theta.
And the moment you change the theta, you have a new policy, therefore you cannot really reuse the sample trajectory tau anymore, and therefore you will need to regenerate new data. Okay, so this… think about the question number 4 and where it is located inside this kind of for loop, because at some point, we will address this kind of problem in this kind of baseline algorithm to move into a direction
Where trajectory and… trajectories, and therefore training data for an algorithm are being reused.
And that kind of reuse will, point to some kind of complications. Our attempt to become more sample efficient will point to some complications in the calculation of the objective function of the… in the PPO algorithm.
Okay, so this was basically the basic kind of reinforcer, policy kind of gradient kind of scheme, and now that we have some kind of idea about the policy gradient, we will bring the two together.
value and the policy, grading kind of methods to implement the PPO. Okay.
So, let's see.
So I want to start,
I think I mentioned the four stages last time of the
Did I mention the four stages?
Of the reinforcement learning when I did the Monte MDP, or probably I did not, so…
Probably I did not in this class, so let me just, write now the…
four stages, of training, if you like, and LLM. So, the subsequent discussion is, titled.
RL fine-tuning, and it's going to be the last thing
That we will treat in this class.
Again, out of scope for the final exam.
Okay, so RL fine-tuning. So, obviously, we are after a very long training kind of exercise.
training. We are coming up with a so-called pre-trained,
a model.
the pre-trained kind of model that knows how to generate the next kind of token fairly well. It has seen trillions of tokens, potentially, as training kind of data, and the whole operation lasts for a long time, and it's pretty expensive.
Obviously, companies have little use for this kind of pre-trained kind of method. The only thing that they're interested in is to supervise, fine-tune
and this is, of course, called, in the literature, SFT, the model, the pretend kind of model to be able to
Address, to answer, questions, the questions that the users have.
Okay, so, effectively, you can,
Supervised finding to, to also do the…
sort of follow instructions or answer questions, and this is basically the border between supervised kind of methods and reinforcement learning. So this is where reinforcement learning kind of starts, and typically involves two steps. One is the
preference.
preference.
fine-tuning.
Okay, the preference fine-tuning, where effectively, over here, we are trying to create, if you like, a model that not only is able to
Sort of,
follow instructions or answer kind of answers, but these answers are, sort of preferable, answers by humans, okay? So, so the preference kind of fine-tuning is what we are using
all the time when you look at ChatGPT, but after that initial kind of release of this type of chat kind of interfaces, they have gone ahead and they have invested quite significantly time on reasoning.
Fine tuning.
Not only the models are able to answer questions, but if these questions include some form of math problems and things like that.
then they are also using, if you like, specific benchmark type of data sets. They are training them with reinforcement learning to be able to solve these kind of math problems.
And, so I showed you, I think I showed you already, an exercise of, what is called the GMSK8K, right, from hiding phase. And in this kind of data set, you have, for example, this type of,
problems which are…
Okay, you can claim it's kind of a simple reasoning kind of problems, and finally, out of this, you get a response, and if this response is correct, then you need to somehow
sort of motivate the agent to respond to take these type of actions more, just like what we did with the kind of policy gradient. And if the response is incorrect, obviously you have to sort of move the thetas in a different kind of direction.
Okay, so what we have, effectively, in our kind of LLM kind of fine-tuning is we are starting with some kind of initial state, S0,
This initial state is the prompt.
But, and, as we just did, earlier, then we obviously have, using, a P, of theta, policy, and this pi of theta policy is the so-called, pre-trained
I don't want to call it pre-trained, I meant to say,
Anything in the, kind of, a border between, supervised,
let me call it SFT model instead of pre-trained.
Okay, so this is basically the policy of the supervised fine-tuning, or the model at the output of the supervised fine-tuning, that will give us an action. Obviously, this action is going to be,
The selection of a token.
Of out of, potentially hundreds of thousands of tokens, or let's say 100,000, tokens is our vocabulary kind of size. The moment I have a token, I will migrate into a new state.
S1. The new state is, the prompt.
plus… The first token.
So I'm migrating to this kind of new state, and the whole process will repeat again with another kind of action, again, using the same, you know, policy.
And obviously, the moment I have a token, I may… before I migrate to this kind of state, the environment… mind you, there's no environment here. That's an important kind of distinction. We don't have a stochastic environment, in the sense that the moment I have… I'm taking an action, I'm selecting, if you like, the token.
the state that I will go to is given. It is this state. It is basically the prompt followed by the token.
However, there is a possibility of receiving a reward.
And the reward could be engineered in multiple ways. One way is to say, you know what, I don't really… I'm not going to give you any reward for just generating a token, but I'll give you a reward at the end.
So I will continue, if you like, this, kind of,
Trajectory, all the way to the point where
I will have, out of some kind of an action, I will form,
the last token, which is, let's say, the number 72 that you saw kind of earlier, and at this moment, I'm going to give you a reward of 1. All other rewards, would actually be 0.
And, this is basically the terminating state.
So, I'm rewarding, Based on the… whether or not the answer is correct.
Let's assume that this is the…
design, if you like, of the reward,
that we would like to introduce here in this kind of fine-tuning exercise. Some, some, some other approach could be, you know, as you can see here,
the model responds with some kind of a reasoning kind of steps, and someone may say, you know what, I'm going to give, for every reasoning step which is concluded, I'm going to give some reward if the step goes in the right direction.
Okay, that's a different reward, but for keeping things kind of a scheme, but for keeping things simple, I'm just going to do the reward at the end.
Okay, so, so out of this kind of reward, you know, so…
the, you know, RT.
I will now… Need… to update.
that… Theta, and therefore the policy.
And obviously, we will have, as I said, some kind of a combination of policy gradient kind of algorithms and value, algorithms. And I think now that we have sort of introduced this,
sort of LLM kind of way of acting, and the need to update a parameters kind of theta, I would like to actually to go back and sort of start introducing what we call a baseline.
So… I want to introduce the concept of a baseline.
So I want to introduce this kind of concept in a way that it is kind of intuitive and not necessarily bore you with a lot of math, although we will get to the math, because the baseline will be, sort of need to be introduced formally into the algorithm. So let's assume that I have two students, student one.
That that takes grades 80, 85,
95, you know, 98, and so on. And student 2,
That takes 8, grades of 50, 60, 55, you know, 45, and so on, in exams, success exams.
And based on the grade of the exams, let's assume that the two students are brothers and, and sister, or whatever you want, combination, and goes to the parent, and the parent gives them a reward every time that they are coming from
You know, an exam… back from an exam.
And if the reward is, of course, based on this absolute kind of test score.
I would, claim that, there's some form of unfairness, so let me see if I can…
Let me see if I can… Je pouf.
suggest the following kind of scenario. Suddenly, this guy, student 2, brings back a kind of a 75.
And, this guy continues to bring back, you know, 85, whatever. In other words, the parent
I think, to be kind of fair, looking at this kind of transition, that the score has improved significantly from student to infers that the student put a lot of effort and started to get better grades. So…
what I'm trying to introduce here is the concept of, let's say, relative reward, and the relative is with respect to what we call a baseline. Okay, so, the student, one, did not really
has a certain baseline and continues, let's say, to perform according to the baseline, but another student had a baseline, which is kind of a poor, and suddenly starts to behave with a better baseline. So there are two things actually going on here. The first one is that, rewards
Can be… Assigned.
Let's say to, Sort of actions.
that… improve.
Relative.
So, improve.
Improve the score.
Oh, come on.
Improve the score.
Relative.
to baseline.
to a baseline.
And… And so, the second kind of implication is that the parent, you know, the patent.
Which we'll call critic, very soon.
Needs.
to be able 2… Adapt.
to baseline.
To, to, not to baseline, to new baselines.
new baselines.
So the parents, seeing the improvement of student kind of 2, will reward the student, let's say, with some kind of money or whatever at the present.
Because it improved compared to the previous baseline that they were. However, as the student goes back into… goes up into 75, 70, 80, and things like that, the new baseline is being established, and the parent will not consider the previous kind of baseline for the reward of money.
But you will need to consider the…
current baseline that they are, right? So, does it make any sense, this,
Does it make any sense, this, this kind of a baseline thing, and this kind of example? And any questions on this?
Okay, so…
Effectively, we have, two things actually going on here. We have an actor, basically, this is the actor.
Actor, so the student is an actor, and we have the parent, which will need to play the critic, and the baseline is… will now be connected to a value
function.
of being in this kind of a state. And, the, actor will continue to act. And what we have seen earlier in the so-called policy kind of gradient kind of method.
we will now need to be revisited, because now the returns will be, relative to a baseline. Okay, will be measured, relative, if you like, returns, rather than absolute returns. I mean, that's basically the bottom line of this,
Of this scheme.
Which, of course, pre-existed PPO. It is part of what we call the
actor-critic type of methods, and more specifically, the advantage actor-critic methods. These are relative.
difference between the returns and the baseline is going to be title and advantage, okay? And that's where the term starts to make some sense. Okay, so I'm going to define here, the advantage.
at time t, And I will define it as A of T.
is equal to… R of T.
minus V.
phi, of SD.
Okay, so this VFi of ST is the…
well-known to us value of ST, and this will be
value of status t, and this will be our baseline. And, and so the advantage is going to, be…
learned.
And the value function is going to be learned out of a different network, so this is the reason why
there is a 5-parameterization here. So there is a critic, if you like, network, and there is an actor network that both of them are performing certain things at the same time. That's what makes the algorithm a bit more complicated than the GRPO that other companies kind of start following after that.
So… So let's write down what we said. For a given
state.
SD.
an action.
AT?
If… The reward.
exceeds the critics.
expectations.
Then… The implication Is… That the action
performed.
better.
Than predicted, because it's a… better than predicted. I mean, if there is a… You know,
there are many, I mean, this is also what people are doing as well, right? Obviously, you are expecting
Something, and, something else happens, and, expressions like, You know…
keeping low your expectations. Some other expectations are in the kind of a common life.
But, so if, if you are… if you understood this kind of, discussion, then, we can actually, suggest that a PPO now, the…
Which is entirely based on an earlier kind of algorithms, to a large degree.
calculates… the advantage?
And, and uses it.
In the… a policy gradient.
Loss.
So, the policy gradient loss, I will call it as L theta.
is, summation.
Let's say from T is equal to 0 to capital T.
Log pi of theta, AT.
And, this is also known as the actor loss.
The value function the value function.
Which is the, baseline.
is estimated.
Using a loss, LFI.
which is, summation over T.
of, v-Fi.
of ST.
Minus… CT?
squared.
So, if we are to,
train a network, which, at the end of the day will be a good predictor. That's, like, let's say, the Y-hat.
For a new state, ST.
And we have… obviously, we have kind of some kind of a training data that will actually give us the returns. Then we can actually estimate in a mean-square error sense, therefore it is kind of a regression problem.
the value of a given kind of state, and we can actually predict it accurately, the value of this kind of… of being in this kind of a state, the state being the concatenation of all tokens generated from the beginning of the formation of an answer up to this moment in time. Sometimes this time is a terminating kind of state.
Okay.
So, so this is basically the two components of the PPO algorithm that we have to bring together. And there is, if you like, some pseudocode that I can
I can actually go through, in order to conclude, if you like, this kind of discussion. This pseudocode is, but before I go to the pseudocode of the PPO, is there any… any question on this kind of baseline advantage and the two networks that we are…
Considering.
Okay, so… So, in the PPO kind of pseudocode.
I will also send you,
a kind of a video, a YouTube video from someone who is, kind of going through a little bit more
with a little bit more… better way, I think, through the PPO, kind of, as a scheme, and what preceded it. So it is definitely a kind of a better treatment that you will get here, so that at least you have some kind of a reference before I…
managed to put my thoughts, at least, in my website. So, again, this is just for information. So, first step is to initialize
The pi of theta end.
the… the network, the parameters of the network which produce, if you like, the WiFi.
Okay, so, for, and then we generate, again, episodes.
Zero to whatever, there is, a large number of episodes.
We collect, form a lot of trajectories, so collect… Lots of trajectories, tau I.
Okay?
By executing.
By theta I.
Okay.
Alright, so… And then,
I'll come to that. I'll come to this kind of line extensively, because initially I'm going to write the algorithm as it first appeared, and then we will…
need to…
revisit what we wrote, and so I'm going to leave a gap over here, just to, put one line that I need to, and I'm going to discuss now the actions, of the PPO. Okay, so we compute
Discounted.
Returns.
GT.
to calculate… L Phi, the critic, loss.
compute… Advantage?
80.
based.
on… Wi-Fi prediction.
And
Update the policy.
based on… the gradient.
theta, L, theta. Now, you know where this is coming from, it's a policy gradient kind of scheme.
And then, update.
the value.
function.
In a similar way, Based, in other words, On the gradient over phi.
of LFI.
Okay, so…
Updates on the parameters, with this kind of gradients are happening for both actor and critic.
Okay, so this algorithm will actually
work fine. I mean, obviously, it's at a high level, but there is definitely… the moment you change the theta, you change the policy, so you have to get rid of the trajectories.
And this becomes, as I said to you when I was presenting the reinforced album, fairly inefficient. And therefore, we are trying to… what we are trying to do here is we're trying to keep these trajectories, okay? Keep these trajectories in memory.
And go ahead and revisit these trajectories, over multiple epochs. So for… Epic.
0, 1, 2, whatever the number of air quotes are. I'm going to revisit these trajectories over and over again in a similar fashion that I'm revisiting training data over and over again in supervised learning. However, there's going to be a problem, because these trajectories were generated with some kind of policy.
And the moment you are
moving away from this policy by updating the theta, the… effectively, the trajectories are… cannot be reused, okay? And if you recall.
Since this kind of discussion, part of the…
I will call it the objective, if you like, function, is to… that we had, if you like, some kind of an expectation inside this objective function from the policy kind of grading algorithm. So, let me write this down.
You know, the… we need…
to find… a way.
Is that… trajectories.
That have been… Generated.
Can be reused.
A gross.
Sita… Updates.
Sure.
actor is changing, trajectories stayed behind, I cannot really use it from one epoch to another. I have to either get rid of it, and stop using epochs and regenerate trajectories, or becoming sample inefficient. And, if you… if you recall inside the policy kind of gradient method, in this
A few moments ago, we had, An expression.
That, involved an expectation.
Okay, so this was basically an…
we had an expectation of a function, okay? So we had an expectation of a function, and there was a gradient in there, right? This is the reason why we did this kind of log gradient, kind of log derivative kind of trick. So.
We need to understand how,
And this gives the pretext on a technique called important sampling.
we're gonna… we're going to do that. We are going to be able to reduce the trajectories. So what is important sampling?
In important sampling, we are able to calculate an expectation.
under… One distribution.
Let me call this distribution P . I'm going… I'm using general notation, using samples.
From… from another distribution, oops, from…
UFX.
Okay.
And, you know, important sampling was initially kind of invented to address the problem of, you know, I have a very complicated kind of P , but I need some statistics out of this P . Let's say a first-order statistic, like an expectation.
And, how can I use another simpler-to-generate and simple-to-simulate kind of probability distribution, Q of X, in order to achieve this, right? So that was the…
Archetypical kind of usage of important sampling.
So, but over here, we kind of, probably you got already where are we going to use this. So, but let me just introduce it, and then we'll see how we're going to use it, because that comes into the specific loss function of the BPO. So this, problem here is to calculate the expectation of F of X
when X is coming out of a distribution P . So this, by definition, is, let's say, summation over X, P . You can write it with any integral as well, P , f of x, and this is the definition of this kind of expectation.
then I can take this summation over X, F of X.
And I'm going to divide and multiply with Q of X.
And then I'm going to have summation over X.
I'm going to write this thing a little bit differently.
I'm going to swap some kind of terms.
P of X divided by Q of X.
F of X.
times?
QFX.
Okay.
But this one is nothing else as the expectation over P.
B OFX.
The OFX.
AirForfX.
Okay, so this is, a kind of,
trick, that we are able to play. They're called a second trick, important sampling, that will allow us to reuse these kind of trajectories, because these trajectories, when they originated from,
PIOLD, or pi-old kind of policy. As the policy becomes pi new, then we are able to… and we need an expectation in the Pi-new policy, we are able to go back to these old samples and calculate that expectation using them.
Okay, that is the reason why, you see this kind of,
this kind of important sampling inside the PPO formulation. Now, with this kind of important sampling, the loss function of the actor
that we are optimizing, in this kind of pseudocode has kind of changed, and this action, this loss function becomes, now incorporates the ratio. As you can see here, it was P divided by Q,
P of X divided by Q of X. The role of the P of X is the pi theta
New, let's say.
And the… The trajectories were generated with a pi theta old.
And, this ratio is in the literature called RT.
RT of theta, okay, in the… you will see it in the paper as RT of theta.
And, obviously we are left with the advantage. You may have noticed, you may have noticed, that,
the log… So we had the… Had some kind of a…
log of the policy, in there, so I… The log was omitted by OpenAI, so the… the log…
was omitted.
by OpenAI.
And…
I do not know why, but the logo is definitely omitted, and you will not see it in the, kind of.
final kind of version. Maybe they have found that it doesn't really add that much to it.
But I do not know the reasons fully. So the final kind of version of the loss of the PPO kind of algorithm will actually have
remind you that this RFT shows how much the new policy changes, right? How much the… so I want to reduce the rate of change of this kind of policy, so I will introduce some kind of a trust region.
And this is the reason why TRPO was preceding the PPO kind of discussion. These trust regions were already part of this kind of discussion.
to… so that the new policy, this, goes inside that kind of a region that I want. So…
I don't want to move… to make significant changes, if you like, so the… the… Final expression…
Oops.
clip.
of, 1 minus epsilon, comma.
RT of theta, comma 1 plus epsilon.
A of T.
Okay, so this, this is, effectively… Contains… Not contains, but constrains.
the policy change.
To happen in this region, 1 minus epsilon, less than or equal to RT.
Of theta less than or 1 plus epsilon.
Or if theta is even smaller than this, then we are picking it, okay? So…
If the change is even smaller, we are maintaining the change that is introduced by the ratio of the two policies.
So this is basically the kind of final conclusion. There is expressions that you will see in the final kind of paper involving IKL divergence.
as a kind of a regularization term, but I think if you understood this, let's call it the canonical PPO kind of scheme, and
And, examples of this kind of PPO scheme, specifically for the, optimizing small language models, is what I'm advising you to study after this kind of course. You will find some examples like this in the Hugging Face, in the Hugging Face website.
And if you cannot find it, just let me know, and I'll point you to this kind of direction.
Okay, so, again, whatever we discussed today, I'm not sure how useful it is, but it's out of scope of the final exam.
And, anything which we discussed up to and inclusive yesterday, last week, it's gonna be in scope.
Any questions, any, any, any comments?
Okay, so I would like to thank you for attending this kind of course. It was a kind of a long journey. I won't be in the final, because I have to be in another exam at NJAT.
As they schedule it right on top of it, but I will, I will have the TAs there, and I'll make sure that the TAs are fully aware of the solutions before they walk into the class, so they can actually guide you through, throughout this, exam. And, most likely, the exam will be easier than the midterm.
I will advise you some topics 48 hours before the… The exam takes place.
in discord.
Alright, thank you.
Ojas Gramopadhye: Thank you, Professor.
Audio shared by Crystal.
Thank you.
Philippa Scroggins: Thank you.
Audio shared by Thank you.
Thank you.
Amrutha Shyam: Thank you, thank you.
Katherine Zhuo: Thank you.
Audio shared by Thank you.