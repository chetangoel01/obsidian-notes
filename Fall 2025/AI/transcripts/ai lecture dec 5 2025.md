Okay, so, welcome to the lecture before the last lecture. So, I'm going to, finish, today the markup decision processes and, reinforcement learning, to some degree, if I can. And if there is a spillover thing, I'm going to do it next time.
And then I will conclude, with some, so by next time when I come here, I will have already developed the final exam.
So I will know, more or less.
what are the questions? I'm not going to tell you, of course, but I am going to provide some kind of review, maybe. You want to review or not, really? Because I don't have a… I've kind of run out of material, unless I start expanding into other classes that I'm teaching, I kind of done with the whole material, so this class was,
An easier class than others to progress a little bit, faster than on average. And,
So this is what I'm going to do. However, before I start, I want to make sure that I ask you about, the pending,
assignments. There was an assignment, was it an assignment or not? Assignment 4? We completed it last week. We completed it last week, okay. Have we received a grade? Yes. Yeah. Okay.
project. The project is due… Sunday. Sunday, okay? I… I'm hearing, all sorts of, stories. It's actually very difficult to have a very wide distribution of experiences in the project.
from the disastrous, I cannot do anything because I have a kind of machine problem, to I'm done, okay? In this kind of spectrum, I advise you, whatever the situation is.
try to… Act as if, you're working, and your manager gave you an assignment, right?
And, you cannot go back and forth constantly, right? Up to a point, I will help you, but after a point, you will need to also do the work, okay? So, unless you're facing some really serious issue.
Like, for example, who is facing now a serious issue? Like, a very serious issue with respect to… they do not know what to submit, they do not know if they're going to submit, what they submit, what they're asking. Is there any question, like, clarification type of question on the project?
Right? So, the pressure was given early enough so that you have a lot of time to ask these type of questions that you are asking, especially in the last two days.
So, let me offer an advice with respect to the knowledge gap, and most of it, I think it has to do with the knowledge gap.
Who is using Nano, no, sorry, not Nano, what is this called? Nano, Nano GraphRab, yeah, okay. Who is using that, thing?
When you're successful in creating a knowledge graph with it? Yes, people are saying yes, it's fine. Make sure you recognize, as I said in my message, that this knowledge graph is not what we are after, right? This knowledge graph has to be augmented.
With a graph that it will
also encode dependencies between concept 1 and Concept 2, like a prerequisite dependency, right? So, although the knowledge graph encodes some form of similarity between concepts, right, they do not necessarily encode that I have to treat this first before I go to the next topic, right?
And so, that is probably the most distinguishing part of a tutoring agent.
And of course, the full-blown tutoring engine will actually also test, evaluate, and things like that as they go along, and then they will proceed only when evaluations are good.
That is missing, of course, for the project, and we don't care about that right now, at least.
Any questions on the graph rack?
Okay, so… so that's fine. Alright, so let's now move on to the…
What we did there last, last week.
The last week, we started the Markov decision processes.
Marital decision processes, why, why we started this discussion?
Actually, it was not last week. Yes? I have a question about the project. You have a question about the project? Just for clarification, what is the deliverable in the whole budget like? What is a deliverable? Yeah. The GitHub repository, together.
With a demonstration of answering these three questions, and attaching the subgraphs that this system used to answer them.
Okay, so, will that be, like, a video of me, creating the, I mean, you have a UI, and you don't have a… either way, CLI, UI, like, web UI doesn't really matter. You can take a video out of it. Better is to clear the screen.
Produce the question and the subgraph, and then…
Take a screenshot of the answer. Okay, so is the subgraph expected from the response of the LLM, or is it, like, something in the back end? Either way is fine. You can prompt it to print the subgraph, or not.
But, definitely we need it. We want to see it.
Yes, so we're using Neo4J for the… for displaying the graph itself. Okay. And that creates, like, a cover and displays a graph with a font of, size 2.
Right? We need to see the… Yeah. Like, oh, this is a graph, you know? Right. What I meant to say is, that creates, like, a dynamic graph that you can click on and everything as well. So, should we just take screenshots of it and add it to our DEADME or something like that? Wait a second.
the graph system, out of the thousands of nodes of the graph, use the specific nodes, and the specific edges, right? Yes. We need this.
Okay, so, so, just a screenshot of it, right? Well, if it is visible, that is fine, yeah.
Actually, there were multiple milestones that were defined, in the course website. Okay, so the second milestone was where we, pick up the text from the web URLs, or the PDFs, or, the YouTube videos. So, we couldn't run that on a local, so, for the assignment purpose, we used four apps.
For the YouTube video. Now, for the project that we're doing, is the expectation that it's written in a, you know, a Python server, or we write a backend application that does this on the go, or just we can just submit for the second milestone, just a notebook that is extracting, text successfully and putting it in module?
Well, okay.
For the second milestone documentation, the second milestone could be, some form of, it's an ingestion pipeline, so…
I'm not so sure if we do need a lot. We just need to see that this thing
you know, in a notebook kind of setting, it does the ingestion logging and output of the ingestion. So you don't need a live, this thing where an application, I'm entering a YouTube video or something? You can, you can do that, but as I mentioned, you know, to you.
Especially this time.
where we are in the deliverable, make any shortcuts you need to make in order to deliver. That is basically the…
In every job is a requirement.
And so…
trainers are learning, etc. I hope you learned something out of our project. There is a significant, Spatial.
significant space for, improvement of this, and, if you watch the website after you're done, I think you…
You will find some enhancements. Okay, so,
Back to the question. Don't say any questions about the project anymore, that's it. Why we started, why we started?
Because we like to torture ourselves, no?
Because, we are aiming in understanding some principles behind fine-tuning of
large language models. And, okay, so we started with a lot of decision processes because the specific fine-tuning we are studying, which is fine-tuning via reinforcement learning, requires knowledge of NDPs.
Bill.
I hope you do remember the Indiana Jones analogy, and you go slow into understanding it, and not crushing it. Which means that you need to spend some time outside of this lecture.
So, the, The system, right, the environment is some kind of initial kind of state.
And an agent decides to act with a specific kind of action, move the system
to the action, move the system to a new stage of spraying, and immediately receive the reward. And this thing continues, and the experiences that we are getting out of it.
are going to be somehow encoded into, one major probability distribution that we call the MPP dynamics.
Okay. And this MDP dynamics is, this probability distribution.
Obviously, we are dealing with a stochastic environment.
The stochastic environment is responsible for the transitions from S to S prime, and also, we are dealing with a stochastic policy.
And, the Stochastic policy is going to…
Stochastically produce sample out of the stochastic processes… stochastic process The specific action.
Okay. That may lead us to the desired state, or not.
Okay? That is… that is basically what this, property distribution is encoding in its full glory.
In MDP, we assume that Good.
In RL, we will not.
This has a significant impact on the algorithms that we will be using to come up with what? What is the question here?
What is a problem statement?
The problem statement is, give me the pi star.
Give me the optimal policy. Give me the way that I will be acting in this environment, such as I will…
Obtain the maximum possible utility of it.
Yes, fair enough at this point. Any questions at this point?
We encoded, in a formalistic graphical model, this equation over here. I hope you recognize the relations between the two.
And we generate it out of this MDP dynamics.
The two models that we will be concerned with…
And we are moving into a trajectory where we are going from one definition to another, hoping that at the end of the day, we will understand two foundational
Objective functions.
Because at the end of the day, the pi star will be obtained just like anything else. The theta star, the W star, whatever star.
From an optimization problem.
So…
The transitional model was defined as a marginalization operation, similar to the reward model, was also defined as a marginalization operation, using the sample.
We spend some time understanding that there is a model in a stochastic world, where with some probability go into the desired state, and some other probability we don't go into the desired state.
Mind you, these grid world environments
are on one extreme of complexity, the simplest possible environment you can use to understand the principles, and on the other extreme, you have the full-blown LLM.
And, make sure that you understand the full-blown from understanding this, and then making the analogies,
Later. Okay, later meaning, today. We'll make these analogies today.
So…
Despite the simplicity, there is some kind of light at the end of the tunnel. So we define the reward function.
That, can take, to, two, two forms.
One is…
I don't care where, where you ended up. The moment you took an action, I'm gonna give you, from that kind of state, I'm gonna give you a reward. And the second form is when I do care.
I will, sort of give you a reward when you transition, take an action, transition to Sprime.
And so this reward function is all the money with respect to how we will ultimately control the behavior of the agent.
You know, so the, the reason why it's the most single, more significant problem,
relatively unsolved still in AI today is that, he has a synonym called evaluation.
Okay.
To the evaluators, Is the one who is going to look at the output of the agent.
And send a reward.
Encouraging or discouraging these type of functions?
So, the environment has the evaluator inside.
In practice, in practical implementation, you have the evaluator kind of sitting as its own kind of class.
And you may have multiple evaluators as well, involved into judging what the agent generated.
Okay.
So we… We have to define a quantity that will capture The following.
I'm not only looking to gain now by doing an Action A, but I'm hoping this action A will lead into states that are promised
To bring me to a better situation.
So make sure that you recognize the translation, that you are able to do the translation between the cohesions in English, and that's a good, test that you can do to test if you understand the thing.
So we did this, quantity.
We call this quantity GDP as a return, right? And, Easily, Sharon.
A random variable or not.
Oh, boom.
Moving on.
Okay, I was drinking water. Okay, so, yes, is a GP a random variable or a deterministic variable?
And the answer to that is.
Well, evidently, he's going to be… Random variable itself.
Why? Because there are two factors of the problem here, that it used to have this data, which is inside that G of T expression. The action is random.
or out of a probability distribution, or completely wrong, but out of a probability distribution. And the state that I'm arriving is also
Potentially not expected one, and so it's random itself.
So, this kind of randomness.
claiming that I have, sort of, some premise by moving to some kind of a stream, introduce into the future randomness in the rewards that I'm expecting to receive.
So… So we can, think about these returns.
Awesome.
Okay, I don't have the graph, but I'll plot a graph a little bit later.
as a discounted summation of all these rewards I'm expecting to receive. Now, unexpected to receive, expectation is always with us in terms of,
setting an optimization problem statement here, right? There's no point of optimizing anything in life If it is instantaneous.
No one is optimizing. I cannot optimize my policy if I'm predicting the stock market by looking at a statania's price. I have to look, just like in supervised learning.
many examples, and optimize supervised learning, we also had the expected risk there, right, as an optimization statement, right? So the same thing here, we will have expectations.
I hope it is no surprise from that.
Okay, a few words about the discount kind of factor. I can, with the setting of the discount factor.
modulate if… The agent is going to be motivated from short-term or long-term benefits.
And finally, the policy.
We always need to think the policy as a probable distribution.
That defines the probability of taking an action from being in state S.
So if I have, let's say, in the green world example, 4 possible actions.
I will expect to see a four-element discrete probability distribution.
And, this is what I have, and if one of the elements is 100%, then I have a deterministic policy.
Any questions up to this moment in time?
Okay, so up to this point in time, everything is kind of smooth, but now we have to come back to that kind of expectation.
And, define… The two, what we call, value functions.
The interview last week, or it was last week, yes.
There was a guide, who…
left OpenAI and went, created his own company, Major Kai. I forget his name always.
Starts with capital S. Yes, that's right, that's okay. Give an interview, it was last week, I think.
And, this guy is not the best speaker, right? Interaction-wise, so… but anyway, you can get some kind of a truth out of this kind of vast experience. He said, you can live with this without these value functions, he said, right? But it will take forever.
You can optimize LLMs without these value functions, but it will take forever.
And there's some kind of intuition to that that I hope by the end of this discussion you will also get.
So the value functions are going to effectively Allow us to…
Estimate, that's basically what they represent, the value of the business in some state. That's the first value function.
Okay? The fact that I'm here.
And if the task is to exit the door, as I said last week.
The value of this kind of state is, lower than the value of being next to the door.
So, if you imagine this room to be a sequence grid world, right, every single sort of cell will have a value.
And, what we are trying to get out of this kind of values is maybe there's a hope to find a way to act optimally by looking at
this…
setting of values. Imagine that you have grids with values, plotting point numbers, in other words, next to you, right? Well, if the cell next to you has a higher value, maybe I can
Moved here. That's it.
That is not, the… A very difficult concept to understand. The maths are a little bit horrendous.
And, the proof I hope you saw it, of the Bellman expectation equation that allows us to say
Well… Here's an Excel… Has this kind of value.
Can I…
tell me… tell anything about the value of this cell. So I'm going a little bit of a backwards way.
In this kind of equation, and
relate the value of this state, S, to the value of the next state. But it's not only the next state, it's only one next state.
And this is a very important thing to understand. In…
markup decision processes, I know the MTP dynamics. So, I can go, as we'll see in a Python code very soon, and I will
Be able to expand.
what we called earlier a backup tree, and which the backup tree is some form of representation of what is really happening. It's very useful, because you can actually understand what is really happening by looking at the picture, and this picture says.
If you're a state of candlefish.
Throw as solid notes, all possible actions you can take out of his state, all possible, or, let's say, up, down, left, and right.
And for each of these kind of actions.
You can actually draw what future stage you can get… you can reach.
Because you know the transitional model.
You can actually calculate the prevalence of each one of them.
Right? And also, you know the reward model, and you can also plug in the reward for each one of them.
This nice situation will never happen to us again.
just doesn't happen in the direction we're going to understand how LLMs are playing.
Which means that if I can expand from this, I can expand from this, I can expand from this.
And now I have a whole range of… future states.
The Saturn's book has a very nice picture about what I'm just describing right now. I'm just going to show it to you, quickly.
If I can.
I'm in the wrong…
There's this picture here.
How would it be?
allows us to full… pool with… there's like a breadth-first kind of expansion.
So, because I… I can know exactly the probabilities and the rewards that I'm going to get from these states, and I can…
go another step, and I can do the same thing, and the same thing, and the same. I'm arriving into a situation where I have
The knowledge translates to a much quicker way of estimating the value of a function.
The value function.
As compared to the reinforcement learning.
Which, the diagram will look quite different on this.
Okay, this is what you need to remember to contrast what's happening in the enforcement.
Okay, fine. That's on board.
So we said, okay, fine, can we solve a problem, a prediction problem now, that would allow us to estimate the value function of a state in a grid-warm environment that, we know everything?
Great.
That was the way we stopped.
That's what we'll do now.
Okay
Today, the 6th.
He's a fifth? Yep.
That's good news, because 6 is my answer.
name dish, like, what's a bad word. Okay, so the…
So, we want to, ask the question… remember, over here there was a plus one, and over there was a minus one.
And, we're asking the question about this state.
Actually, the target.
I want to estimate… Goodbye.
Pretty sweet.
It is.
I'm expecting to see, if I go back to the formula, and to this kind of backup tree, I will be guided by the backup tree, okay? I will be guided by the backup tree. So I am in,
S33.
And I'm drawing the backup tree in order for me to hope that, I will be able to use the other values of the states,
And we'll see now how the value… the values… there's a chicken and egg problem right here, right? Because I do not know the values of the other states, right? We'll see how the chicken and egg problem is solved. But I… at least I will be able to express the V by S3
as a function of the values of the other states I can potentially transition to.
So, I'm, I'm growing, I'm growing.
Oh, by the way, I want to, out of the four possible actions I will be able to make, I want to just expand the…
A is equal to right.
Okay? I'm just going right only, and if you understand that, I wouldn't understand it.
So, immediately from this action item 2, I can transition to… Pre, in fact, other states.
because I'm having a stochastic
what I call PSS prime for short. PSS prime is that stochastic transition model from S to S prime.
So… If I go right, I will arrive at S43.
With probability 80%.
I will, but I will also hit the wall and stay at S3 with probability
10%. And, I can transition to SV2, With another profit of 10%.
Okay?
So I can write it now, as I'll get some… Immediate reward?
Gamma?
B.
of S43, given S3D, comma, A is equal to right, times, times repi of S.
Or three.
Plus.
plus P.
of S33 given S33 comma A is equal to right, times V pi of S33.
Class B.
of S3, given S33, Comma A is equal to right.
We buy for S32.
Are we in a… in a good, place?
So… I did this for the S33, right?
I can go and,
bring this revised 3 on the other side, and do all this kind of manipulation, but bottom line is not this, the algebraic manipulations I can do, is that I have now an equation that has not one unknown, but has more than one unknown.
But, if I go ahead and do this… exercise.
For the other states that are involved in this equation.
I'm going to have now a second equation.
With, even more unknowns, because I will transition to other states.
For example, if I expand here, I can transition to stay here, because I'm gonna hit here, go here, or go here.
If I now take any other, S primes, this one, for example, I can do the same. In general, how many equations can I draft?
11.
How many unknowns in this equation as there will be?
11.
Are these equations linear or nonlinear?
They are linear.
This guy… Yes, sir.
0.8.
This guy… These guys are both… Beautiful and well.
So I have a system of linear equations, with… The same number of unknowns.
So, back to elementary school, you did the solution to that using a matriculation.
So I can solve the system and get… My value function, Completely, for all the states.
So let's do that using, some kind of a simple
Example first, before we go into this kind of, like, legal world.
What is it?
Okay.
This task that we are… doing.
Answering the question, what is my value function in this, in an environment?
It's called the prediction problem.
And, or policy evaluation.
And, the policy that we're evaluating over here, we tried to evaluate the previous example as well, could be any probability distribution you can think of, right? It could be a uniform random policy. One fourth, probability to go left, one fourth probability to go right, one for probability to go up and down.
So over here, we have an example which is simpler than the grid world. I'll come back to the grid world after treating this example.
Where the policy is the deterministic policy, and there are only two states.
S1 and S2.
And, an environment is as follows.
the dynamics are a force. If you're in S1,
Taking an action with both ways, too.
Which probability 1.
And?
You get a reward of 2.
We are in S2, the same axi, there's only one axi here, leads to S1, With reward of zero.
Estimate the value function of this system.
So, first of all, the transition model, the PSS prime.
I hope you recognize that this is the PSS path. This is a transition model.
is… Transpose identity matrix.
Which I probably want to go from one state to another.
I hope we organize this is the reward vector.
2 and Z, and 0. S1, 10 is 2.
That's, that's, the reward is, that's one comma.
S, S1 comma A, comma S prime, right?
2, and the other one is 0.
And this Bellman equation, expectation equation that we have looked at, right, It can't be implemented.
It can be… it can be translated into a vector form.
I have a value vector. This value vector would have as many elements as the number of states.
I have an immediate rewards are… plus gamma.
capital P, which is a transition model, times the other value vector.
what was the other value vector? The other value vector is the value vector of S prime.
be of surprise. So… I can take this, and I can say, okay, I have,
I have now a way to solve this kind of system by bringing this term over here.
Calculating i minus gamma 3 pi is equal to R, And then I have…
A system that is, just like I did it earlier for the grid work, a system of two equations with two unknowns.
Right? And, this is the capital A. A times X is equal to B.
Right? That is a system that, like…
I can solve the system.
For the X.
And I have a vector of D1, it will be 10, and 3D will be 9.
Which, of course, the moment I have a value estimate such as this, right, and I have these rewards as I assign them, you know, the environment assigned rewards, we can suggest the following kind of optimization of the policy. We will do that later, but I'm just telling you right now.
If you are an STO,
Does it… does it take to go to S1?
Yes, it does, because the value of S1 is larger than the value of S2.
If you are an S1, Stay in this one.
There's no point to go to a store. I will say that greedy approach of acting in an algorithm is called policy iteration.
That solves, interactively the Markov decision process problem. Yes, go ahead. Could you please explain that again, because, like, you said that seeing it as one pays more, but, I mean, isn't revert or S2?
Are there?
Now, the value out of this system, right, we calculated the V1 value, the value of S1 to be 10, and the value of S2 to be 9.
So, that's a little non-intuitive, though, because, like, I mean, the value function is also, like, a function of the rewards, so…
No, no, but given the rewards, given the rewards.
I'm telling you what reward you will receive. I'm not asking to design a report function.
I'm telling you how it will function.
I'm feeling a little while.
So we acknowledged.
In this kind of a setting, there's no degree of freedom.
Of course, you can go ahead and, assign,
Gamma and things like that, yes, go ahead.
Is there a value for staying in SMO, or taking the action at SMO?
Okay.
I kind of, cheated a little bit here when I… I translated what we see here in,
in projecting how to act optimally. Over here, we have one action, and we don't have the option to stay in the same state. But if we had…
that possibility, not in this kind of realm, but in a bit more difficult kind of problem. We could select the optimal action by doing some kind of ability way of acting, by moving into states that have higher value.
Okay, but over here, we have deterministic going… the moment you are in S2, deterministically, you will go to S1 with S probability, and you will receive the reward.
Yeah, scrolling.
The moment you are in this one, action leads the S2.
So, isn't that my reward to move out of this one? To go to this room?
This one.
gives you a higher reward, yeah. Yeah, actually, it's two and reward us too, that's right, that's right. And, you have a probability of,
of moving into S2 to reward of 2, and from S2, action leads to S1 with reward of 0. Yeah, so what's the point? The point is that the value… the values are not, reflecting the intuition. Yeah.
I need to go back to Caesar.
Okay, there is a… Okay, for this specimen factory.
Excuse me.
Yes, the moment you are in this one.
Right? Because you are deterministically going to S2, only with this action, you are going to get a reward. So the value of being in S1 is larger than the value of being in S2.
You are going to move to S1, probability 1.
So, can you please give an intuition of what the value exactly means? Like, what does it exactly say? So, the value is…
Of a state is an estimate of future rewards.
Okay, so it's, like, the future rewards. Future Rewards?
Starting from that stage, right? But that doesn't mean that we should… we benefit from staying in that state, right? Again, I…
that I put a bit more… I put a bit more into the problem than what the problem statement says, okay? I'm saying, in the grid world, as we will see, right, when we have the ability to
multiple things in a state, right, to select the proper action, I will apply some greedy policy to move optimally, to states of better value, that's all I say. But that's not a… it's an extrapolation of
To this problem destination.
It's an example.
Because of this kind of matrix inversion, I told you, every time you see sufficient matrix inversion, what do you do?
you know, don't implement it, because it more likely it will blow up.
We need to do, we need to solve the same problem with an interactive method.
So the iteration is… the basis of this iteration method is
That, realization that the Bellman operator is a contraction.
What is a contraction?
I'm giving you a very simple example.
Of a scalar quadraction.
And of course, the Bellman is doing a vector-based kind of contraction, but if we understand this.
We have this kind of evacuation.
XK plus 1.
Where is the… if you do this, where the X will converge?
So… For gamma, which is less than 1.
So you can actually do this.
or C is equal to 2.
And, understand… that the X will actually converge to
C divided by 1 minus 1.
In the case of infinite iterations. So, that is a contraction. We have…
We have some kind of a convergence to a fixed value.
It turns out that the Bellman operator is a contraction, and therefore it can play the same trick, and avoid the matrix inversion, which, as I said, is dangerous.
2?
Allow me to estimate the fee.
It's sort of an action.
So, I will do that. What is contracting again?
Contraction means that we are, no matter where you start with, right, in this X, right, you will see that this X will convert into a fixed point.
This is,
common to many, which is a common realization to many, we don't do, any…
by any stretch of the imagination, any argument with respect to convergence here, right? We're just realizing that the equation is amenable to iterative implementation. Convergence is something completely different in our story.
So I went… I went ahead and I wrote the Wellman expectation equation as an iteration over here.
And, I will apply the same method.
Duchamp.
The problem. Again, the trivial to state problem I gave you. I'm starting from some initial theme.
I am iteration 1.
I'm implementing the equation.
the output, the V1, is now becoming
This V is called venue. This venue becomes reord here. Create a new venue.
this being involved here, upgrade, and so on and so on. So, we are, over many iterations, we will…
Right? We'll progress to the V to the Y function.
That is not a question.
These are all numerics that helps us kind of understand what's happening with the Bellman expectation equation and how to solve it.
Right? How to… Estimated, not solid, estimated, that means predicted.
Obviously, we can do the same in the mini-grid example that we just had. We drew it. We realized that it's 11 equations with 11 unknowns, and we need to now solve it with this, with this equation over here.
Me neutrado puede.
I'm not writing these equations down, please make sure that when you review, you don't look at the notes and,
You know,
Yeah, I write it. So I need to write it down, because it's an important equivalent.
Augustine?
Not with this.
But iteratively.
Excuse your book.
As follows.
Fair.
3K plus 1.
Of this is equal.
Summation over 8.
Y of a given S.
Summation over S-blind chromocar.
Pi of S prime, comma R, given S comma A.
POFAR slash VK… of Sprank.
this equation… It's known as the… Interrupt Policy.
evaluation.
Or, the biodiversity.
We'll see now the Python code to, that evaluates this, value action for this big work.
Did you finish writing it?
Okay.
Not a specific… Meaning, reader board, whatever we'll call it.
that I drew. It's a slightly larger picture. And it's all borrowed from,
some Python libraries, which people typically use to,
Use to understand what's happening without
having to face thousands of states and stuff like that, as I told you.
So, this function over here is,
Obviously has the environment, right, as an argument.
a specific policy that I would like to evaluate. I'm evaluating a specific policy.
I'm assuming a comma?
And I'm assuming, some epsilon that will help me escape that infinite loop.
Obviously, that happens to me
If the value function, in other words, is not changing more than epsilon, right, I'm stopping the iteration, and I'm reducing the re-match.
All right, so we are building, out of this kind of environment, my transition model. I know it, my reward function, I know it, right?
And, I'm starting the for loop.
I'm going to set the… 3 to all zeros, just like I did earlier, with the two-state thing.
And, I'm going to start Sage.
while true, that's anything in the loop that I will break up depending on epsilon. For S is in the range of the number of states for all S's, so I'm effectively doing this for all S's, right? This…
Populate, for each action.
For its action, which, of course, an action is determined by sampling out of this policy, Right?
And I'm returning, out of this policy, the action and reaction probability, which I need.
Per what I wrote earlier in this, fundraising panel evaluation.
And for each action, look at the possible next states.
It's the fact that I know it.
It's a part of the noise that allows me to write an equation Which is…
It replays a new renewals.
Right?
Based on… that…
value, so there are two things actually happening here, okay? Calculate the expected value. So, then, first of all, there's a… remember, there was a plus equal sign over here, right? So, if you go back to the backup diagram.
for each state, I know where I can transition, right?
each of these states is offering me a value, right? And I can use this value
in an arrow, which I will call Bootstrapped from now on.
It's a very, very common term in this domain, right? Bootstrapping.
That allows me to, sort of say something about this from the values of this.
Sure.
What are these, interactive equation is saying, summation over A of the probability of taking an action, right?
Dives.
the…
So, okay, so this is, basically the art.
plus VK of S prime, okay? So… What, what is it?
So… The hero we have.
The action probability times the transition
probability to go to this S prime of R, Plus, Gamble.
The office time.
Google Fishback.
The next, the real vest next issue on here.
Are you seeing the habits?
And the classical, here is the summation.
And I'm doing it for all essays.
Doing it for all this history.
So… I will set that now in the outlaw this…
S, I will set this to this, V of S is now this, V, vector regulated over here, the V divided regulated over here.
And I'm sending it to the corresponding vector location of the corresponding state.
And I'm doing it over many, many iterations.
And checking against the cash flow, I'm stopping, and producing the value.
If you do this and do it step by step.
You will arrive at the value function of all 35 states.
And this is where I told you that at some point, we may want to
act greedily, and go from one state, if I know this, I can go from one state to a state of a better value.
But… There is a problem there. Because… My actions are stochastic.
I mean, in general, the actions are stochastic.
And we'll map the actions to…
an LLM later on, right? It's a stochastic device.
So…
it's better to enhance this value function, enhance it in a sense that maybe I can include in the value function the action that I'm taking.
And if now I have values of… if you validate this and take the action to go here, that is the value that you will end up.
If you take another action and you end up here, that is another value that you will end up. This type of functions that take into account both states and actions are called two functions.
There are no quality functions. I'm not, I'm not going to, anymore, from now on.
call state value function, because it's a mouthful, or a state action value function, and we will call the third one V function, and the second one Q function.
Okay, so I'm going to refine now the Q function.
This is the second definition of the objective.
State?
Auction.
value, function.
Professor Egg.
Dubai, or first banquet.
is equal.
I hope no one is surprised by expectation operator.
RT plus 1.
Class drama.
VPI?
of ST plus 1 is equal to S.
Brian?
Given?
ST is equal to S, and… AT is equal to A.
As you can see, the conditioning is there.
To say, hey, now you are committing to this action.
Obviously, in probabilistic sense, For the specific action now that you have, I'm all conditioned on the action, right?
no longer, probabilistic quantity.
Oh, this is going to be… equal to… Summation?
Okay, Spran, coma R.
Thief, of their spine.
Comma R, given S comma A.
R?
plus gamma V pi of S prime.
So cute.
see what's happening now in the backup tree. From S…
I can take executing the policy pie.
many actions. Each of the actions I'm gonna take carries a value.
And for whatever leak, State that the known transitional model will lead me to, right?
These next states will have Values, as well. State values.
The equation allows me to calculate a Q,
based on the V pi of S. But it's better, better, if I'm calculating the Q based on, in an interactive way again, based on an earlier Q.
So, I've needed another equation over here.
Is that a world bought?
write us…
Off and close as my agenda.
Nem agenda books.
And the second equation is, Prepare office.
It's summation over A.
of time, of A, given us.
Sure.
Which means that… this new equation.
Let me, let me not draw this aside, let me draw the last one.
Which I know to make a point again.
Can we expand this guy?
Each of these future actions, A prime, As the QPI,
of S prime, comma A prime.
Correct?
So… These two equations allow me
Can we hold that much stopping.
In other words, if you see what's happening over here… We buy a fresh grind.
Is… Summation over A prime.
pi of a prime given S prime, Q pi of S prime, comma A prime.
So I have now, on the right-hand side, a Q pi of S prime comma a prime, and on the left-hand side, I have a Q pi of S comma A.
Therefore, I have related.
the tool?
Brunch.
QH.
Therefore, I call this bootstrapping.
I'm entitled to call it bootstrapping in the same way as I use bootstrapping in a Bellman expectation operation for the V function.
That is a Bellman expectation operation for the Q function.
This Q function, however, is… Better!
Compared to the refactor, every time that I want to act optimally, because I can now know
What is the value you can extract by taking an action, which is really the definition in the direction of acting optimally.
I select the action that gives me the better value if I transition.
If we reach this point, an understanding of the full week's backup.
pair the Python code that I just showed you, right?
Which, of course, depends on the knowledge of the transitional model and the reward function.
And we understand the V function and the Q function here. The rest is…
When we look at the reinforcement learning, we'll see that the rest are approximations to what we just discussed, okay, in terms of estimating these varying functions.
I don't have… I think I have a queue function called in my website in the same… similar way.
But I would like now to go into… okay, fine, let's assume that I am able to predict this function, the V function, the Q function, right?
In this kind of iterative kind of way. What is the…
What is the… what is the…
What is called now the control.
So the problem has the prediction stage, which I'm evaluating the functions, and the control stage, where now, given the fact that I know the function, I'm able to go to a better policy.
This control part is solved with two algorithms, which are iterative in nature, and it's called policy iteration and value iteration.
For the interest of time, I'm going to just go through policy duration, okay, just to understand what's really happening, and then I'm going to go to reinforcement learning, okay?
So, now, all the previous kind of discussion, Is… can be titled… Objective value functions.
And…
prediction.
problems.
The second headline over here is… Control?
We are, let's say, policy, iteration.
In order to study the control via this algorithm and others, like value iteration.
We need to understand two more Behrman equations. The…
Two Bellman equations we have to understand are called Bellman ophthalmology equations.
And they are going to be nonlinear.
They're gonna be non-linear, because… As you might expect, The gritty way… That I'm acting.
will introduce a max function in those previous equations, which are max functions are obviously nonlinear operations, right? So I cannot really solve it with a matrix inversion or with the previous kind of method, but there is hope at the end of the tunnel that I'll be able to solve
this MDP, finally solved it, come up with a pi star, in other words, now. Which, of course, meaning coming up with a pi star, I will go back to my word over here, and I can actually start drawing the arrows that are optimal
Actions in every single state here.
And it could be more than one arrow in some states, right? For example, in this state, it could be an optimum, it would go here or here. I do not know. The pi star will tell you.
So that's what we're trying to do with the controller.
We, we know… What's happening in terms of value?
in a real queue.
And here we are going to the direction of coming up with a pi star with a control.
So the two equations that are called
Bellman optimality equations are going to be.
This is probably the most difficult part.
I want to start the discussion by… kind of organizing.
what I call now the fish star.
office.
And before, I had VPI.
Now…
I have this star, or sometimes I'll use it as pi star, it doesn't really matter. And the star is always, used as a notation optimization theory to denote the optimal thing, right? We know that.
So, Max?
over A.
of Q, pi star.
of S comma A.
This…
Let me spark up to me.
Is what? Oh, there is an overlap with, interesting.
Oh, what happened?
No, no, no.
I am sure.
Let me show up here.
I'm sure.
from here.
George.
Stop showing.
Okay, do you see anything?
Okay.
Okay, now it's better. You can see here…
It's true start of this command.
It's not a fiscal year.
The equation. It says something extremely simple.
Which is basically what I just told you a few moments ago. If you know the best possible
Q function that you can extract from the system.
In a states that we can transition to.
Go there.
Yes, go ahead. What is the pi star vertical? Pi star means optimal force. Yes, yes.
This is… this is optimal policy.
is… obviously, we are trying to estimate the optimal policy, right? But this gives us the intuition of what we're trying to do. I mean, how to connect the
possible value that I can possibly extract by being in this state S, right, is related to… via the backup tree, again, to the Q star, where the previous equation, or the definition of the Q star.
It's vice versa, right?
So this is going to be… okay.
This is gonna be…
Max?
over-policify.
Probably pi of S.
It's equal to… max over policy, over, over, over A.
of expectation.
Over pi star.
for GT.
Given ST is equal to S, comma AT is equal to A.
Is equal to max.
over A.
of the expectation.
5 star.
For parti plus 1 plus gamma Vista.
Or festivals 1.
given ST is equal to S.
NAT is required to pay.
It's equal to max… Over 8, summation.
Alright.
Sure, not R prime, but S prime, comma R.
P.
of S prime, comma R, given S, comma A.
Art Nas gamo, V-star.
We'll press play.
short bracket.
So let's translate this kind of equations in English.
First, translation.
the value the volume.
of a state.
under… And optimal.
policy.
must be equal.
To the expected return.
For the best action.
Doc?
Thanks.
The other way of saying it is that for each state.
S… We go.
True.
We evaluate, in other words, all… Polish is fine.
And be craft.
the policy?
That… maximizes…
V pi of S, or Q pi of S comma, doesn't matter.
No.
formal… We present the policy iteration as an algorithm.
I want to, understand a little bit more about, you know, to present, if you like, an example of, similar to the green world, but it's
It's given a bit differently.
This example is inside the second book.
And it's called me very much into animals.
So I have a robot who is Asian, obviously, and is collecting empty cans in an oxygen bar.
It has essentials.
And a grieper, nerf, and a grieper that can pick up
And place them in a beam.
The robot is obviously operating, powered by a battery.
And, we can interpret the sensing.
And, we can't understand.
What is happening in terms of the gripper?
And, we have an environment here where this agent would be operated.
Our job in this kind of environment is to command the agent to change
Actions, like, for example.
Depending on the battery state. So, if the battery is low, maybe the agent's better to go back to charge itself. If the battery is high, maybe the agent will actually go and pick up more cans and stuff like that.
So… So, we have three options, search.
Where the agent will move around searching for the recycling kind of can.
Remain stationary, and wait for someone to bring you the can. That's another action. And three, get back to the home base to recharge the battery. These are the three actions.
So we estimate via this example, right, the pi star. That's the… that's the job.
Okay, so let's do… let's look at the… Final state machine.
The final state machine.
It's actually very common in embedded systems to see finite state machines.
And, so the example is… the example is kind of, of some value in general. So we have two states, high and low, like we had earlier in S1 and S2.
And, for each stage, we have potentially a different set of actions.
Obviously, when I'm in low, I can…
go back to the chart, so I transition the state high.
We are an action, recharge, and I will be rewarded
By this, with, a reward of…
Does it say any reward anywhere? From an old.
To recharge, to go to high, with probability 1, reward of 0.
Okay, so one comma here, there's a comma, the projector is, not, allowing you to see the comma, there's 1 is the transition probability, and 0 is the reward.
The same thing happens here. If you are in high, you can search.
If you… as you continue to search, you can transition to high, up to a point where
You will be transitioned with some probability to low, because you will have depleted the battery.
It makes some sense. Transitions, I think if you go back home and look a little bit further at this kind of example, the final state machine will make some sense. Obviously, over here have muted everything, we know everything, and we know everything.
So our job is to write down
just like I wrote earlier, the equation for the V pi of S33, right, I'm going to now write down the equation of the V,
star of a state. In this case, I am writing two equations, one for the high and one for the low.
So I have two equations, and I have, unknowns, two unknowns, but unfortunately these equations will not be linear.
equations. There will be nonlinear equations. That is the difference of what's happening now.
Because I will have max operators in them.
Are you following where we are in the… Okay, so that's basically the two equations I'm going to write. I'm going to write only one, and then leave with the second equation for you to write at home when you are looking at this. Okay.
So, this is the recycling robot.
Example.
Or… We start… or fight.
The equation is a goal, right?
The question is a book. The question is this guy.
Oh, I'm coming.
I'm writing.
Oops, I'm writing.
Mox.
Over A, it is, right?
I am in a state sorry, what am I writing? Restart off…
H over the state H, of the state H.
Okay, so, it is much overrated. What actions can we get, from the state, from the state, CHI?
Second task. Wait. Search.
That's it. Huh?
Don't have any other option, okay? So…
I'm expecting to see a max.
Followed by 2… Elements with comma separated in them, right?
So, I'm, writing this.
If you go back home, you will recognize this, so I'm square bracket, curly bracket, the probability of
H, given that I'm in H.
And… I'm doing action of search.
Oh. Ours.
of age.
given… Search.
Comma, H. H is a high state, right? H is there.
Aye.
Battery… State?
Gas.
Los.
D?
of arriving to low when I'm in high, and I continue to search. I hope it makes some sense. Okay, what am I… I'm missing some… I'm looking at a different role here. I didn't write the plus… so it's a reward plus gamma.
plus gamma, V, star.
of low, because low is the next state I can go to, okay? This is the curve… this is the square bracket ends.
Okay? You understand?
Does the next day from the high attendee.
Now, if I continue to search.
while I'm in high, I will end up in low with some probability.
Oh, sorry, with that option. Thanks, given the name?
This is not… this is not low, it's age.
Wait a second, yes, yes, we are, we are K. We are K. P of low, given, and high, continue to search.
Again, R.
off.
Carly. Search.
Lowe?
Class?
Plus…
Plus gamma?
Vista.
of laws.
Nope.
There's a comma now.
Because… The first step… The maximums of our actions. How many actions I have, too.
So, the first action was search.
And the other action from Hyde is to go and
What is the other action?
I forgot. Wait!
The other action is weight.
I'm looking at this, and I'm just writing. Okay, I'm not doing anything intelligent here.
The second term is… 5, comma.
So it's high given, probability of high given, high and weight.
of R, of high given weight.
on our high.
Last gamma, B star.
Flash?
Flash?
P, of low given high comma. Weight.
or R.
times R of high given Weight given low.
Comma low, plus gamma V star for low.
This is where the close bracket ends.
So this is the…
First term?
And this is the… where I am.
awakened.
two options I have, two terms, separated by… Come on.
This is equivalent to the first equation we wrote.
earlier, which is… was linear, right? Obviously, this equation now is nonlinear.
And I can write another equation again for the V-star of low.
The Vista of Lowe, if you go home and do it, it will have 3 terms.
We don't have 3 terms.
Because it has the options.
Of course, there's also the action of Recharge.
But we need to find a solution for that.
This clarifying rules.
Now, and we will do this.
By policy direction. That is the algorithm that I was talking about earlier.
There is a light at the end of the tunnel.
I'll go a little bit quicker now in terms of policy durations, because we are, in some shape and form.
Kind of understanding what it really, this is,
one thing before I go to policy duration.
is that I now have optimal values, either V or Q. I didn't do the Q, but it's similar, okay? I have now optimal values, either V star or Q star. And obviously.
I'm gonna go… Ability action.
no matter where I am in my policy, I can do a great reaction, and I can transition to a better policy.
Because now, I know where to go.
And this control, signal control kind of scheme could be evaluated, unfortunately, Over here, I have the…
This, this, the same,
I have another Python code, but I didn't print the output, so I'm going to skip this, I'm going to skip it. I'll do it tomorrow and tell you a little bit about it, right, next time. But this is basically the premise of the policy iteration kind of algorithm.
You know, start from some value, And some polish.
initial policy, right? You do not know where you end up.
You are trying to find a pi star. You end up in the pi star. You evaluate the policy first, with the same method as doing the policy evaluation.
And… You are applying the greedy control.
That I showed you just now. And after many iterations, after many iterations, you will end up with
a V star, and a pi star.
That is the premise.
startled.
The contraction operation.
likely for us, is still with us here, even with the Bellman optimality.
Which means that I got some V and I,
I gotta evaluate… there are two lines here, this is all conceptual, okay? There are two lines. One is a value line, and the other is the policy line.
I can evaluate the policy and get into a new Vprime.
the renewal.
I can take the venue, because now I have it, and I can act greedily on the venue to come up with a better policy by new.
And I can then evaluate upon you to come up with a new renew.
and do the same thing over and over again, and convergence-wise, the theorem says that I will converge to B star and pi star.
Now, proof of convergences are inside the silences book.
Okay.
And there's… all of these algorithms are coming under the headline of generalized policy duration. So policy duration is a good thing to spend some time to understand
To understand it and solve it. We have now some examples to help you, kind of, find it for a simple grid work problem.
But, in this kind of a grid wall problem, We have,
I don't know, 14 possible states.
There are two terminating states.
1 and 14? Sorry, is there any one terminal state, some device are shaped square to find?
The action that the agent takes caused the intended state transition with variability 1,
Actions leading out of the gradient state unchanged. The reward is minus 1 until the terminal state is reached.
And ladies and follow us a uniform round of words.
Okay, so we'll start with a uniform random policy, right? And it will end up with a pi star.
That's the… that's the exercise here. Still, we are in this kind of a small world, situation.
So, I'm starting… with a V?
Right? And survived.
As you can see, pi is uniformly random, 3 is all zeros.
And what I'm gonna do, go up to meet the evaluation lines of the policy. I need to evaluate this guy.
how much support evaluated. I'm going to… Plug into the equation, right?
And… I'm… I'm using there.
policy evaluation thing we did earlier, the interactive form of the policy evaluation we did kind of earlier, so I'm liking all these numbers, okay? I'm not going to do it right now, I'm not going to spend with time with one for leaders of time. Just do it on your own, right? Make sure that you understand how to evaluate
this policy and create a new matrix. In this case, it's a matrix of value.
providers.
Okay.
If you… if you come… if you do the calculations, you arrive in this VMatrix.
Are you… it's just plain policy evaluation, right?
So, I'm going to arrive at these values by doing all this kind of stuff, right?
Now what I can do, what is the algorithm that's telling me? Act greedily on the value.
What it really means? Move. Start selecting actions now, before you can do everything. Now, in each cell, you need to select
From that finite set of form.
Because that selection will lead you to a better state. Better state meaning better value, a state with a better value, according to what you see here. Well, evidently, this guy will select to go left.
And eliminate all the other three possibilities. The same thing happens to here, and here, and here. This is all, I think, self-evident. I don't think I need to spend time on this. You can do the same now. You know, you have a new policy.
You have a new bonus?
Go ahead and evaluate it. Now you go back up.
to that line, right? Evaluation of, of… The policy from iteration 1.
How you do?
the same thing.
The same thing, and the same thing, have the same thing.
And at some point, Your… We wanna stop?
And select this pesto.
Visual license.
Now, we'll see that When we do reinforcement learning on
Language models and things like that, right?
Obviously, we do not know the transition model and the reward model, but this kind of,
changing policy constantly is going to be with us, and doing some certain things out of the previous policy and stuff like that. So try to understand this for MDP, and then you will be in good shape to understand what's happening in the reinforcement there.
And reinforcement learning starts now,
So… MDP is… has its own kind of set of assumptions. What we are…
now we do not know whether enforcement learning. We do not know
That transition, and we do not know that it works.
So, the reward model, and things like that, or… so we will need to…
Act a little bit differently in order for us to
estimate these quantities that we need. So… That's sheets.
What's gonna happen?
What happens?
the pencil… he had some allergic reaction to MTPs.
Brain force.
Okay, it's not the writing.
Okay, sorry. No, it's fully charged. I'm a bit worried about, don't worry about.
We have a…
Oh, unfortunate.
See, maybe I can, build the application.
Thank you.
Okay, start at the bottom.
Okay, there are two schemes in reinforcement learning that we will now
when we write now the backup trees, right, will become evident now why we have them, and why they are… they lead to more or less all the algorithms. One is the Monte Carlo method.
And the other is, temporal dysfunction.
Eating method.
So, I'll start with the first one.
Our job in reinforcement learning says we do not know things.
Is the only thing we can do is… generate experiences.
And, the environment will give us whatever reward. The environment will take us to whatever state is prime.
And,
We then need to pick up all this data, effectively, that we will treat as kind of training data, out of which we will learn
certain qualities for us to help us with the act, as we did earlier, optimally, to extract some kind of optimal kind of behavior. So, there is a…
The diagram that we had, kind of, earlier, in terms of state.
I remember it was, lead us to some,
Using this kind of policy pie, we take, let's say, an action, and, give us,
If you like some… reward to take us to S prime, and so on and so on.
this thing… Given in S prime, we'll do the same.
Not necessarily the same action, so let me just… Remove that.
And so on and so on.
Up until…
The interaction terminates.
So… What we have done now, we started with some kind of a state, and executed
An episode?
Okay.
I executed the next one.
maybe remove the charger, maybe it would be better. Maybe soon.
That we'll also call it a trajectory.
Can we generate multiple episodes? Yes, we can.
We will do the same thing again and again and again.
And I don't want to draw another set kind of a note, but starting here from some kind of state S,
I may actually give it some common states, but my, definitely, my second edition will be different from the previous one. I'll keep different states, because I took different actions.
And, I received, of course, different rewards. So…
Can we estimate, out of each
episode, some form of return? Yes, we can, because we can… we have… we are storing all this data.
as we go, we store all the data, and therefore, for each of these kind of episodes, I can calculate the return that I've seen.
Starting with that status.
So, in reinforcement learning, relative to what we have seen, kind of, earlier.
pulled back.
in reinforcement learning, value, what I have seen kind of earlier, We are doing this.
I cannot really go ahead and evaluate, in other words, bootstrap.
with reinforcement learning. I cannot know
all the stuff that will take me here, and here, and here. The only thing I can do is let myself at the mercy of the… my own decisions and the environment.
Therefore, I'm doing depth First type of… Generation of the data.
until I terminate. And another episode of me here, and maybe here, and so on, so I am…
Constantly generating these kind of trajectories, and…
Obviously, because I'm making some states from my trajectory, and maybe I will see the state again in another trajectory.
I'll go to that state again, maybe.
I can start counting.
how many times have been to that state, and what reward I sow after that kind of state, right? And I can do the best
unbiased estimator I can ever invented, I can calculate the sample mean of those returns.
if I ask you, In other words, what is the…
value of being in this kind of state. Clair in the previous equation in MDP, there was an expectation. Now, instead of an expectation, I have a sample mean of future rewards.
Future discounted rewards, or a sample mean of returns.
So, I have, let's say, these two trajectors here.
This state over here.
Come on. This state over here, was visible.
Twice.
Because I only see two transactions here, right? So I will convey what happened after that, because I…
I was given the information, it's in my data, I'm storing the data, and I can see, okay, I got the one return from the yellow, and another return when I was in the magenta trajectory, and for this state S double prime, I have now, I will add
The G.
yellow, and G magenta, and divide by 2.
I'm following. Nothing, nothing peculiar is happening here. I'm just replacing these expectations with The shampoo me.
Okay, so let's write it down.
Fourth.
Beach?
Time step.
E?
at S.
is visited.
The site is visited.
If… an episode.
Thanks for the mad.
the counter.
Anal fish.
This is basically we're keeping up track of how many times I've gone to that state.
Then we can calculate.
The total.
Return?
that we… How? Seeing?
across… Episodes?
Chrome.
of that state.
G.
tau, not capital T for total. Of S is equal to G total of S.
Lash GT.
At the end.
intelligence.
on average?
V pi of S is equal to D of S divided by N.
I'll press?
And… As… Whoa.
Us.
NFS goes mainframe.
this quantity.
will be approximating the V pi of S.
Because the more the episodes, the higher the probability that I will approximate the same estimate as a full backup allowed me to estimate in dynamic
Programming or macrovision program.
You know, earlier kind of thing. I was approximating the food backup by generating multiple, multiple episodes and doing this calculation of relative.
Very intuitive, right?
So… And very simple. Now, in terms of an equation, however, we want to write this equation of,
Coming up with,
sort of VIP IOFS as an approximation from the sample means by revisiting what we have done in carbon filters.
Many moons ago.
At some point, I was looking at a drone localization problem, and I was telling you.
That that location on that kind of line can be written
that estimate, from, as I'm actually estimate, from the sample mean, right?
Because I had the Gaussian random variables, remember? So I was… at that time, I was sitting…
over here, I was typing this equation, which I called at the time incremental, Meme?
For example, Meeting?
Approximation.
I was writing Munecade, is equal to 1 over K, summation.
change equal to 1 to K of XJs.
I think we had measurements there with basically, Z or something.
So then we have 1 over K… of,
Let me not derive it again. You can go back over there and persuade yourselves that this mu of k is the mu k minus 1.
plus 1 over K.
affects gay.
minus mu A minus 1.
I think you should remember that equation.
At some point, I was telling you that this 1 over K, factor over here will later be called Kalman gain.
So, effectively, I wrote that the new sample mean estimate is the previous sample mean estimate, plus
The carbon gain of the new information Minors are pretty much just…
So, when you go to textbooks and you see this,
Monte Carlo equation, you will see this equation, which matches exactly what we have
Pro just now. You will see… I'll write it exactly underneath.
reply of this?
is equal V pi, S, plus alpha, GT minus, we bind fresh.
Where alpha is…
1 over N of S, and I told you what it is, the number of times that I visited state S.
Thank you.
value of the status will be the previous value of the status, plus the gain factor, right, times…
The new information had arrived from this episode.
The new information that arrived on this episode is the return.
Minus the previous estimate of the time.
This confusing equation has its genesis over here, what we discussed many times. So, this is what you see in the textbook.
And this one you will see.
A version of this now, you will see also in next week when we do the…
LLM, PPO discussion. Approximal fault systemization discussion.
True.
Cool thing, too.
Retrieve out of this kind of discussion.
I have to terminate in order for me To tell you the Jesus.
How would that be named? That's a major limitation.
Major limitation.
So, the reason why it's a major limitation is that in many instances, it's okay to wait until the car is crashing, and to go back and say, okay.
I have now an estimate of the value of not doing a sort of 100 miles per hour in a corner, right? And in reality, I prefer to have schemes that allow me to act as I go.
And so, this is a prefix to the pretext to the discussion of temporal difference methods.
The temporal difference methods will look similarly to the Monte Carlo method, but they will allow us to act as we go.
Before we go there, Your, core site has another example from Savage's book.
where I was given, again, an environment.
This is called the Markov Rewards kind of process.
And, obviously, I can write down
all the equations, right, with the corresponding kind of unknowns, to calculate this black line here. This is a value function.
prom, the lack of reward process.
If the… if I knew everything, right? If I do not know
anything right now. What I actually can start doing, or what I call alpha earlier, right, for each of these different alfalfas, I can plot
What? Right?
error is, the empirical error, right, between the ideal
value, which only a full-blown backup tree, right, will allow me, the one in the MTP process, right, will allow me to do. But with sample returns.
Using the equation I just drew, I just wrote down, the Markov, the Markov's chain equation.
So, as you can see, I'm generating many samples.
And, For different forgetting factors, there come slightly different behaviors.
Forgetting factor will play some kind of role. This is, by the way,
Point 4, I have no idea, I cannot even read.
Yeah.04.
This is 0.04, this is .01, right? So, this is, the forgetting factor, which is…
closer to zero, right, will converts, obviously.
Later, right, than a forgetting factor, which is larger, closer.
One, for example.
So this is the behavior we expect to see, seeing among the dogs. I will learn, but obviously I need many episodes to…
and learn the value function. And as we just looked at.
Learning the value function is the first step for me to act optimally.
In a race for a living or setting.
Now, the temporal difference method is, it's as follows.
While the Monte Carlo method was the equation above, and temporal difference
of zero method, I'll explain the zero in a moment, is Be proud of.
Office leads.
is equal to V pi of ST.
plus alpha.
RT plus 1.
That's gamma.
V pi of ST plus 1.
Minus d pi.
Christie.
So, this term over here, is that… Only thing.
That changed from the Monte Carlo equations. You see your notes, over there, we have… we had the sample return.
In a Monte Cargo, right?
So… Here, we have this term, which is actually called a TD target.
And basically, The TD target, If you look at it, does it remind you anything?
It's, like, one step, bootstrap.
from, remember what the sample GP was trying to do, right? Sample Z was saying, okay.
for this episode, you know, that is a sample return. That is basically the discounted future reward that you expect to see, right, until the end of the episode. That's basically what the sample return was, conveying.
Over here, it is as if we attempt to do some kind of bootstrap.
That reminds us what we were doing in MTP.
how we are actually able to do good staffing and still, you know, in a world that we don't have all the information to do good staffing, this is the thing that we need to understand a little bit, right? So, let's,
Let us hide it down.
Go one step…
one step because of the… the way when we are… wrote the TD target, obviously in
A more complicated version will come. The one-step, TD of 0,
combines.
the bootstrapping.
dynamic programming.
And?
the sampling, off.
MC.
This… Is there… secret behind the DD of 0.
I cannot do this, definitely, right?
But I will act as if iPad?
The value function of this state?
The value of this state.
to footstrap over the sample… trajectory.
That connection is delivered in your state.
That sampling trajectory was generated, obviously, by… in the Monte Carlo map. In the Monte Carlo map, we're doing this all the way to the terminating stage, right?
Are you following, right? So, it is as if you're doing all the car load, but…
You are updating the value before waiting for the terminating state to arrive.
So, you are doing this. Not waiting is manifested on the fact that
You are effectively telling, you see the equation.
The equation says, I have an idea of what your future returns will be. I don't have to.
just to wait until you give me the GT, because the GT is only when you terminate it.
I have a VPIO 50 plus one.
It may be around VPI of V pi of ST plus 1, but I will still use it.
will I converse to something? And the convergence, sort of, proof for DP of 0 says, yes, you will
That's the goal.
premise of TDO0, and that is a… an elaborate version of this, PPO was entirely, based on.
I have to leave at 1.15 today, okay? But I have 5 more minutes, so I will present to you the 10-step version of the TD of 0… of the TD algorithm, and closing discussion with TD of,
algorithm, and then next week, I'm going to start the fine-tuning discussion for… I will apply this to the fine-tuning discussion, in other words.
So the TD of lambda.
is an enhancement of TD of 0 to allow us to the couple…
the… Time?
Actions are taken.
And?
Volume.
function.
Updates.
are done.
In other words, what the city of Lambda says, you know.
Don't update me with just one step immediately. Wait.
Until, potentially, you have even a better estimate of the value, and then again.
So, if I am in Brooklyn, I want to go back to Jersey, as I will do now, I have two options, right? I will act to go through the Brooklyn Bridge, or I have to… I can go through the Manhattan Bridge, right?
The TDO says, boy, the moment you cross the Manhattan Bridge.
Go ahead and update the value of Brooklyn.
The TD of Lambda says, wait until you exit the Holland Tunnel and you reach New Jersey before you're obtain the Brooklyn, because you never know what is waiting for you in Canal Street.
And usually, not many good things are waiting in the country.
All right, so that is really the TD of lambda, and I'm going to write down the equation. This equation is a little bit, sort of complicated in some form of trying to introduce certain things of discounting here, but I will write the equation, and I will explain a little bit next time. So…
I'm defining a quantity, sold, GT.
of N, which is RT plus 1.
plus gamma, RT plus 2.
Last… last gamma.
N minus 1.
Arti Placen.
market the power of flame.
Z.
T+.
N minus 1.
S. Did La Saint.
And instead of writing the pi as a subscript, because the subscript now is, is, is,
occupied by the index, which I am going to stop and do the bootstrapping. I put it as a superscript here, the same thing.
So, simply speaking, the equation V pi of ST.
Ezekiel, goodbye. Ophesty.
plus alpha.
GT.
offhand.
minus p pi of ST.
Yes?
the… 10-step equation.
So next time, I will start from here.
And then I will exam the… Some kind of, odd…
an immediate kind of step with this kind of a lambda would act as kind of an exponential weighting factor, and present, then switch to the LLM discussion.
As I said, the PPO discussion, we will focus on at length is going to be entirely based on TPO land.
Okay.
All right, thank you, and don't forget, there is no time, be careful, there is no time to open tickets, for… oh, I forgot.
to upload the URL of my GitHub repository for the project, something happened to it, you know, all this kind of stuff, there's no time. They have to grade ASAP, and the only thing that remains is to grade the final exam, okay? So, make sure you submit correctly your project.
Make sure if you are… did you collaborate in this class for the project? Yeah. Make sure you list your collaborators in the README file at the top.
Thing over there, because there will be a common grade.
No attendance, there's no attendance today or next time.