What happened?
Thank you.
Those are your complaint.
Part of it.
Good morning.
Okay, welcome to the show.
Third, before the last lecture.
This is the… we have only 3 lectures remaining, this one and another two.
And, if you haven't signed, please pass it around when you won't present signed.
And I wanted to start with this assignment that apparently created a lot of stress, right?
Alright, so, so what I, what I was saying.
There was a lot of stress, and realizations of how important hardware is these days, as well.
And, also, I should also tell you, now that we are approaching the holiday season.
That there is a… the stress is going even further up.
and somehow it's also correlated with how Bitcoin is dropping. At the same time, there are negative correlation events. Okay? So, I think the…
The assignment floor is, just to recap, just looking at the tickets that you guys have opened.
I think there was a ticket on, concured usage.
We cannot run it. Some people didn't try Collab and open some secrets before trying. I have heard that from other students as well, that Notebook is able to run in Colab.
No.
In, there should be some tricks that we can do for those who are running it locally to minimize the compute
situation, the impact on the conflict situation. One.
Is that you don't have to process the whole regions.
I don't think there's a problem on task 1, because, did anyone face any problem on Task 1? We get the transcript out?
Yeah, okay. So, the other type of, questions that we faced is,
Okay, what are we gonna do, to feed to the LLM of your choice, right?
the context of the transcript, right? And there are many ways,
You could try to insert it as a prompt.
straightforward way. And also, you can try to insert it as a…
Vector database, which means that you have to check it, and pass it on as a retrieval of when it's generated.
I don't think any other way, that it's, I think this… both of them are… it's very easy to just win 50 points in this assignment. I think straightforward, moving 50 points. Anyone wants to answer some, the questions kind of successfully? Yes, okay, so some people, okay.
So Task 2, this is the one that, created a lot of issues with compute.
we first need to segment the video, right? And in order to segment the video into plays, right, you can actually think about some methods. I mean, one method that we could potentially use is that since you know the players, and since you know the transcript.
the computer really does want, you can infer the timestamps
But you have to chop the video to, segment it into offensive and defensive plays.
So I'm giving you more or less the one good idea, I think. So who has tried that?
Yes, okay. So, we don't have to generate, 20,000 plays. Yes, go ahead. In this task 2.1.
When you're talking about offensive and defensive plays. Which team? We have to pick a particular team, right? Yeah, you have to pick up a player in one team. This is for Task 2, but task 1 does not separate, does not… does not team-specific. Yeah, but, like.
from the definitions there, it's like, we have to consider different movements, so it's not just one player we have to focus, like, for example, triangle offense. It's not just based on one player, so we have to… No, we have to segment a lot of players in the field. Yeah, so it will probably be for one particular team, right?
Not necessarily. I mean, if one team has the ball, the other team plays defense, and vice versa. If so, if one team has the possession.
then that team will probably be attacking, and the other team will be defending. So, then what… what times, like, what do we categorize that sequence? Well, you know, I mean, okay.
You can't segment the team, I mean, you're gonna know the team because you have the list of players.
Right? From the list of players, you can actually associate the… who is having the ball at this moment in time, because they're… they are telling me, actually, right?
And then you can actually do the defensive and offensive kind of play. But obviously, as you said, a play is a play. It could be offensive for one team and defensive for the other, okay? But you have to segment the plays. Now, I don't expect you to take the whole hour, one hour and a half, and segment it all, if you have computer resources issue.
But I don't think the compute resource is on this task.
I don't think the computer should be honest. Anyone segment… manage to segment the video?
Yes, you, in some seconds, each play? I sequentially run the whole, like, I segmented it to the chapters. So, video segmentation is, you know, obviously it is done routinely in, in YouTube, for example, you can see the chapters we have under the video, right?
And in instances, video segmentation is relatively not so easy, but a straightforward kind of problem, because you can use some kind of measure to measure, you know, changes in the scene, right? And therefore detect segments like this.
Or you can, using the transcript, you do detect the segments, right, in the conversation, for example.
this specific, sports kind of game is… the scene is not really changing that dramatically, unless, of course, they are changing the camera view, and they're focusing on, you know, the breaks that they are having, and the, you know, the shots that they're doing, where there's a foul or something. And then,
And then it started kind of a video only will not cut it. You have to use the transcript.
Which is probably an easier task. Now, if you generate,
you know, 50 plays, that's, I guess, okay.
we don't have to generate thousands out of the whole video. I think, not thousands, I don't know how many places in the whole video, but…
You know, probably a large number.
But so, moving now to Casco Point 2.2.
Again, you're focusing on a place where the specific player is playing, and again, you can detect that from the…
transcript as well, since you segmented at least, anyway. And, you can, this is basically where the action recognition is required, right?
And this action of ammunition, you are free to select
A VLM, as we said. Anyone who has selected a VLM that was giving you some… who has done this task 2.2?
All of you, okay, that's great. Okay, so,
So you can select a VLM that will actually allow you to ground, potentially, an action from the transcript.
And also… also, there is a visual component that you can, sort of use as well.
But even if you're using mostly the transcript, right, to detect if you like the action, it's going to be okay, right, provided that you also fill in the table, and the action is…
True, okay? I mean, I don't… since you have the… you have the, sort of, the information from the transcript, I think it's okay.
Especially the situation is, that, that it's due in 2 days, right? I don't know if you have a lot of time to do this in its full glory, okay?
Now, only one player, as I said, but you have to… you have to pick up this guy. You have to pick up the state, okay? And you can pick up the state from the video, obviously, right? You have to state the state of the court.
Okay, do we have to draw the court? Like, if you click that link at the bottom of that page, it'll show you.
Which one? So, this one? The one at the bottom of the page? If you click that, it will load the image.
This guy? Source.
Yeah.
Well, the image is, let me scroll all the way down now. The image that it's,
Is that what you want? Yeah, that's right.
I mean, this one you will get from the… Processed video.
Well, obviously, the process video is the same duration as, what you,
there is no kind of video, right? So it's not, so obviously you can actually go to the specific sections where you identify the action, and chop it, right, and take a snapshot if you like, and the process video has the position of the players, right, in the…
Because they are providing this for us, conveniently.
Right, so I hope the stress is going a little bit down, as far as assignment for is concerned, and it should increase as far as your project is concerned. So, who has managed to work on the project?
Where is the project view?
7th? Well, you have plenty of time.
Yo,
Okay, so that's fine. So you have two… you have two weeks, I guess. You have something like two weeks to finish the project. So try to get rid of this assignment, as soon as possible, and, just focus on,
on, the project and, the back.
Alright, so let's now move on.
I'm gonna do two things today. One is I'm gonna finish this exciting topic of logical reasoning.
And, I want to just, provide some kind of a… Compressed treatment of classical climbing.
Because my aim is to expose her to… Reinforcement learning.
Because I think this one will be a little bit more useful to you, than other kind of topics in this course.
I don't know if you're going to have the full time to close the loop, so to speak. As I told you, this kind of goes beyond… a little bit beyond the scope of the class, but I'll give you some pointers at the end of the day to how to apply reinforcement learning for
language models, visual language models, and things like that, right? So this is… this is definitely a very essential,
sort of element. If we don't have the time and run out of time, because I should tell you, MGP
The discussion on Reform Learning is going to go… from this book.
And this book is,
500 pages, so I'm not going to finish the… all the topics, I'm just going to hit some topics which I believe are important, and I'm going to leave many topics behind. I do advise you, however, because the market is very explosive right now with respect to reinforcement learning, to find the time to,
during the winter break, and, finish what I'm gonna leave behind.
Okay, that's what I'll put now.
Let's see.
For what we did, last week?
Is, we went through,
We went through, kind of, some principles of logical reasoning, and, we saw, two methods that we do logical reasoning. One is the work model checking, which we did.
And the one that we do today is the improving, which works only on the syntactical side, and, with other word models.
Peer improving is,
type of theorem proving, not really a theorem proving we're doing today, but field improving in general, or problematic theory improving, is what I told you AWS is using in their logical reasoning, you know, for cybersecurity.
So, so it's a much, much bigger topic, but I'm just going to bypass the digital parts.
Okay, so we went through the primitive stuff, what is a model, what are the set of models that's fine.
the knowledge base or a sentence in, specifically. We saw some kind of a truth table. We went through this kind of game where, initially, we, moved the agent around, doing logical reasoning with our brain, and, after.
reviewing…
the operations that we need to be doing in the… with the knowledge base, because logical reasoning is entirely based on interactions with a knowledge base. We have two sets of interaction. One is a tell interaction, or the write, and the other is an ask interaction, or the read.
We, sort of started defining
visually, all the possibilities from all these read and write operations that we can possibly have, right, with the knowledge base, right? So, I hope this was kind of understood. We went through entailment, contingency, contradiction, and
We use those, to, sort of, understand when the knowledge base responds with true, false, or I do not know.
Then we said, okay, fine. Let us, now,
Sort of understand how the machine will do logical reasoning with, first, the simplest of the methods called model checking.
And, we went through the… A kind of a query.
About, something in the moves world, and we found a way to respond to it, to this.
Obviously, there's a table that I don't have on the screen, but you know where it is, right? It's in your course site. And enumerated all the possible models of the world, and for those who satisfy the knowledge base, we went there, and
Agility, though.
Show you… Okay.
We went there.
Yep.
Nope.
We went there, and for those…
models. These are all the models as part of the knowledge base. We went and looked what they say with respect to specificity. Now.
If, one row of the specific symbol says, true, and the other says false,
Then, obviously, the response is… is what?
From our headspace.
I do not know. I don't have a consistent response from the knowledge base, right? Some models are true, some models are false.
With respect to this double symbol, but in this case, it was,
Not the case, but in general, it can happen.
So now, I'm going to, go over, the second, The second, method.
And, the second method is, titled… Today, we have, 11-2195.
And, the second method is, It's called logical reasoning.
Absolutely.
the positioner…
Go ahead, absolutely.
The pollen… The whole… the whole method here is done entirely based on, Proof.
And therefore, it will be, based on, two important kind of inference rules that we will define now.
So there are inference rules.
that, will allow us to, sort of, do the inference entirely based on syntax. One is called, modus.
components.
And goes as follows.
It's written like this, where the numerator here is premise, And the nominator is conclusion.
Obviously, it's not a fraction, it's just the way that we are defining it.
Yeah, kind of notation they are defining.
So, if A implies
beta, and we know that A is true, then we can conclude that beta is true. That is more disappointing.
Our job here is to take To start from the query.
And start transforming this query into…
Rules that we know they're evaluating something, true or false.
And, that way, gradually, we will,
So, we're given a query, we start from something that we know from our world, like a rule that we can,
sort of, we have in the knowledge base. And, from that kind of knowledge base, rules, or rule of rules, we will arrive at the conclusion, which is, basically, has to do with the specific way we received. And then we can respond to our faults or…
So the first one is modus ponens. The second inference rule is AND elimination.
Again, the same location.
What I use?
Conclusion.
play and better flow when playing is strong.
And, price bills.
So these are the two arguments that Viola will use.
And together with these rules, we will use the table that I told you to have a printout of.
Which is a table.
7?
7-11th.
A 7-Eleven table, which is a table of…
I'm not sure if I have it here.
I don't think I have it here. I don't have any in the port side, but I showed you last time this table. Now, I have to find it, though, because I need to use it.
Here's the problem.
Logical?
refunding?
I should put it in the cross-site at some point. No physical reasoning, chapter setting.
Not chapter, Table 7-11.
7, 11.
Okay.
This is the table.
Okay?
This you have to bring it to the table.
So what the table is telling you is, if you have something on the left side, it's equivalent, you can add it to what is on the right side, and this is where we will progress on proving that this query is either true or false, okay?
Alright, so we will… we will start now.
So… This is an example. So.
that.
P12.
S.
B.
2-1.
He's true.
Well, obviously.
We're already given the answer that the knowledge base will respond here, but obviously the query is on the left-hand side of the
This is a sentence over here. This is… this is our query, right?
So…
This query comes from the Hubus World problem that we did last week. The Hubus World problem had a whole range of sentences and rules in the knowledge base.
Let me go back to the Unbooks world. This was the situation over there. We had a whole bunch of rules, and one rule that I would like to… so you look at the query, and definitely you see symbols P12 and P21, right there, being involved.
And,
you can actually start with a rule, that you have in the knowledge base, and I'll start with rule R2. You can start from any rule, it makes sense. I… obviously, if you start with a rule that does not have these symbols, probably the…
attempt to prove it is not going to be very successful. So you need to start from something which is… and this is basically a more or less kind of experience, rather than anything else.
So…
I'm going to start from Rule R2. So I have the rule R2 in front of me now, and I'm going to write it. I'm going to write it as…
Hard to think so.
Scott?
Rome.
R2… Which is… B11?
double implication, P12, or P21.
This is what we have in the knowledge base. I hope you remember that if there is a breeze in some location, this mist around it, it should have some pits.
Okay.
So, I go to the… I go to the, shot of,
lookup table, of, equivalences, and, I'm going to… Replace the double implication with P11 implies P12.
or P21.
And?
P12.
or 321 implies B11. I hope you agree.
I will call this R8.
The ruling it.
No.
And I'm going to go back to my 7-Eleven table.
And there is a line here.
that kind of matches what I just wrote. Which, which line is this? There's a name. Every line, every line has a name. Which name, which line you think I should look like?
This is the bike… what I did is called biconditional elimination, right? This is the line I just applied here, so I want… I want to write these things so you… you write these things down. This is basically the…
by conditional.
Elimination.
This is what I need.
Alright, so… I'm looking out at the inference rule called end elimination.
Right?
And, and, and elimination, so this end elimination says.
with a hypothesis, that this is R8 is true, I'm going to… Right?
P12.
4P21, implies.
V11.
Look at the end elimination over here.
we have A, alpha, and beta implies alpha, but exactly the same thing, we can write alpha and beta implies beta.
Right? So I did… I didn't write you this, but, it's equivalent to…
to that. This is basically what I did. And this is basically end elimination rule.
I'm calling this R9.
Then I'm going to go to the table.
to the… to the 7-Eleven table.
And,
look at your notes, I cannot really show both at the same time. I have a P12 or P21, implies B11.
So, I will… it's not evident.
the rule that I'm applying here is…
the rule of contraposition, this line. This is not evident, okay? I admit that, it's not evident.
But, if I apply the zoom.
What is the point composition is suggesting?
Well, I'm gonna write down the…
Evolution of that with the rule contrapposition.
Look at the 7-Elevens.
not… Better implies So, not beta implies, not at A. That is the… that is the route, right?
So, what I have here, what is my beta? Nord beta.
Implies what?
Nope.
P12.
or P21.
This is contraposition.
Oh, I wrote it already.
We know this.
So, I'm gonna call this R10.
Now, another, application of the first, International Road over here.
They're modest foreigners.
involves sentences.
our port.
And R10.
What is the sentence R4 and R10? R4.
It is, it's coming from the knowledge base.
Where it's an output app above here. Let's look at the R4.
4th. 4th.
is not B11?
Right? This is the R4 here. Do you see that? R4 is not B11.
So…
So, R4, strong knowledge base.
So, I have… R10 also here.
and R4, so I'm saying, not B11 implies not P12 or P21, comma, not B11.
Can you see that?
That's the… that's the…
premise for modus pollens, so what should I conclude? What should I conclude out of this?
That… Better, right?
So, what is the beta here?
This guy. So I'm concluding… Not P12 or P21.
Huh?
This is from…
Are you okay?
So This rule, I will call it R11.
And now I'm going to go to the 7-Eleven table.
7-Eleven table.
Is what? I don't know.
Which, role do you think I should, use?
The Morgan? Who's on the Morgan?
No one? Okay.
I'm trying to, grab into something to, pretend that everything is, fine here. No? B1, 2, or P.
Dual 1.
is equivalent to not P12 and not P21.
This is the Morgan.
So, I have… started.
prompt… something which is…
true in the knowledge base, and after a subsequent kind of steps, either with covalences or inferences, I concluded something which is
Proof.
Okay, I don't pretend that this is, the most,
easy example of a syntactical way of proving something. And,
I don't think that, anyone will ask you in the final exam about, to prove something in a syntactical way. I'm just mentioning because these are two major methods, and, there are…
implications, as I said, on,
Going from propositional logic to first-order logic, like what your textbook is doing.
first-order logic is what AWS is using.
So, you can actually do this transition at a later time. I don't have any…
even any desire, honestly, to actually, teach, first-order logic. I'm glad I'm not a logician by any perfect imagination.
Okay. So, I'm gonna stay here. I hope you got something out of logical reasoning, with this kind of two methods, but the journey continues with first-order logic. First…
out there.
logic works. Effectively, over there, we have predicates.
And you have symbols, and it kind of matches, to some degree,
It is a journey to look at the you know, hey, WS, cyber security.
Where they have this automatic theorem proving.
And, or, neurosymbolic reasoning.
That combines symbolic representations such as this, And, neural networks.
So I'm gonna move down to classical planning.
automated planning, because I want to start the MDB discussion today.
So, this was left over from last week, so we're gonna do classical planning here.
Or planning with, or planning without any interactions.
with the environment. That's the meaning of classical math.
The subsequent discussion will always involve interaction with the environment via these reward signals that the environment is sending us.
So, obviously, the field of classical advantage is yet another vast kind of field, and you know, obviously, many of you are
Must have done some, has some exposure, potentially, in the past of,
The transition from some state of the world, right, to some kind of a…
desired state that I want to, to end up via a sequence of steps.
Great. Now… The whole discussion in this kind of case is non-stochastic. Our environment is deterministic here.
And,
how… what is the proper order of, what we end up defining here in a language called PDDL?
is a solution that the search algorithm is providing. So, our job here in the classical kind of planning is to be able to specify correctly the world model.
The domain, in other words, and the specific instance of the problem, And,
Press the solver, just like an optimizer, but it's working in the search… with search algorithms, to… that will spit out these sequences.
did both. Actions.
that need to be taken to arrive to this state. And obviously, this sequence of actions will actually be optimal in what sense? We have the ability to suggest that they could be, optimal in the number of steps, or some other kind of criteria.
Okay, that is the core job of classical Milani. Now.
We… where is classical planning, is, done today?
One example that could potentially be, applicable is,
Every time there is a very narrow kind of domain, like in manufacturing.
You have, let's say, a body car that, is doing…
this thing again and again and again, right? Someone would actually suggest how I actually can optimally move
This product from the…
from the belt to the pallet for packing, stuff like that. Like, fairly deterministic environments where nothing can potentially change because these things are in gauges, right?
So there's a… there's a whole domain that is very useful in this kind of case, in those, applications. Another domain that is actually, useful, to understand a couple of principles there is,
In, domains, that, involve some form of, verification. Like, verification,
the verification is actually a very big thing today. I mean, especially when it comes to AI, because
The whole domain of,
Fine-tuning either language models, or vision models, or, other, all sorts of models, multimodal models, is based on,
reinforcement learning. And although this is a topic that involves, obviously, interactions with the environment.
The most critical part in a reinforcement learning system is the verification that the answer that something came out of the model is correct or not.
So, lots and lots of people today are making a lot of revenue, lots of money, Building environments.
general environments, right? For example.
Don't think about this as an environment, but think about any possible job function you can think of.
Like, from the nurse in a hospital, to a supply chain manager in a pharmaceutical company, to whatever, right?
These people are doing, certain actions, they are doing certain, that includes, sort of, planning, for example, demand planning, manufacturing planning, all this kind of stuff, right? So this kind of algorithms, this kind of tools are coming, in modeling these environments as accurately as possible.
And, these environments, if you have a good world model, if you like, the…
That can, send back the ride.
reward signal. Then the model, that it is very stochastic and noisy and things like that, will start to focus more and more into a behavior that you want to drive into.
This behavior could be very deterministic, or could be something else, right? Or it could be very domain-specific.
So…
when we study things, you know, I think, one mistake that many people are actually doing is that they are looking at the
Latest and greatest, but behind this latest and greatest, there is a lot of principles such as this, which are behind the scenes, which are… needed to be done, for these latest and greatest to be of any value.
Okay, so I'm going to present, in this automatic planning, space, a language called PDDR, Planning Definition, Planning Domain,
definitional language. And, with this kind of language, we will,
feed these, specifications into a solver. This solver is, could be very generic, or could be very specific to the, to, let's say, to… we have various types of solvers, like, just like we have various types of optimization algorithms.
And then we'll get a solution at the end of the day, actually, a response that is satisfies, if you like, the percentage of the problem.
Now, to give you another example, last year, there was an assignment that had students
model the domain of OpenRoute.
What is open round?
website where you can run models? Yeah, it's a router towards LLMs, right? So what you do is, what they do is they receive requests, and they feed it to the most appropriate model to satisfy that request. And sometimes.
Sometimes this is an informed decision, if they have special agreements, like they see the request, and sometimes it's just basically a load balancing kind of decision, without really seeing any requests.
But, obviously.
Behind the scenes, there is a very smart way scheduler, right? Which says, okay, I want to assign this request to that resource, right? So it's a resource management kind of problem that can be modeled with PDDM.
to resolve into, sort of some form of load balancing, for example. So one of the first things that has to be done there is to model the domain of OpenRouter, which means that you have to
Go and line out.
All the entities, the objects, sorry, the classes that open another business consists.
But then this exercise is actually very, very common in a variety of problems. Most of the problems of AI in enterprises start with ontology.
definition.
So when you want to provide a solution to a customer in pharmaceutical, you cannot just,
Say, oh, use a… ChatGPT here, so you can use ChatGPT, but then, obviously, no one will,
take you seriously, right? You have to… you have to go in and say, okay, what is your processes, what is your business? What are the… what are you trying to do? And people are doing ontology by definition, right, first.
Understand what are the classes? Oh, I have, sort of, prime ingredient. So, prime ingredient is a class, right? And, for every, medicine, there is prime ingredient 1, 2, 3, and 4 that have to come together to form a compound, let's say.
And so all of this kind of has to be modeled, right? And we'll start with modeling the domain with PDDR, so we can understand some principles of it.
So, classical planning involves, I won't… I won't qualify with PDDR.
Which is, learning.
domain?
definition.
Lance.
So, the PGDL… involves a specification of two things. One is the domain.
And the second one is… they're prominent.
So, what is really the domain? So, domain?
specifies the types.
Classes in object-oriented languages.
By the way, PDDL is an example of what we call domain-specific languages, right? DSLs.
Okay, so you have, types, you have predicates.
Which is obviously a state.
Something…
about… the subject.
Of a sentence.
You have, action, schemas.
Also known as operators.
Action schemas, you can also, as you see them for action templates.
And, we'll see the examples of that. And the other one is… Precondition.
That need to be satisfied in order for an action to be able to be taken, and effects.
So, just to understand a little bit this kind of language, we are going to be using a domain called Blocks World.
Which is a domain where, we have a surface where we light up, lay, blocks.
And let's say we have a robotic arm over there that is going to take this block and move it to specific positions in order to satisfy the problem, right, that we're going to throw it at you.
So, for example, a problem, In this kind of notch world.
Will involve some definition of an initial state, Where block here is B.
Logs A and C are over there.
That's my initial stage.
Just like the image we saw earlier with the train and stuff like that. And, this… is…
That gold state.
That is their problem.
And there are two files that end with extension PDDL,
in, in this, domain, PDDR.
and property detail that we put all this information and write it.
Okay, so what is really, what is a PDDL stadium?
Make it more formal.
definition.
is a conjunction.
Off the ground.
atomic.
Fluence.
which I want to… which I want you to think of the fluence as kind of variables.
So… how this is kind of translated, i.e, Specific.
grounded.
Arrangement.
of objects.
that, and also, informed.
predicates.
with constant parameters.
So, to give you an example, how a sentence, which is going to be, involves some elements of first-order logic.
Although it's kind of trivial, so you can understand it without really having a dedicated treatment officer or the logic, I have a state led such as this. How do I describe it?
I suggest that this state is.
Involves a predicate on.
A, comma.
table, I'm gonna write the whole word.
And?
on.
Bink on a table.
And… on.
C, comma 8.
That is a PDDR state.
So… Obviously, this PDF state is
Associated with a specific kind of problem, so we can… we will… we'll see that inside the…
Instantiation of the… of the problem statement, description of the problem statement.
Well, we're gonna have states which are the initial states and the goal states, but that's basically what we'll have inside the probability DDL.
Going back to the… Sort of a domain, however.
Obviously, we need to define types, like blocks. A block is not a block. Block is a type, right here, in this kind of world.
And predicates, on is an example of a predicate over here.
And action schemas. Now, these action schemas are not,
straightforward, so I will just spend some time discussing action schemas.
they represent.
the family.
of ground action.
So, the action schema, will actually involve preconditions and effects, right? So,
Let's, let's look at an example. Move.
B?
Call my ex, call my wife.
And, so, for example, B will actually be a block.
X will be, let's say, the table, and Y will be another block instance, right?
And so, so, for example, in the, go in the end.
in part of the kind of gold state,
we have this B sitting on top of C, right? The block B sitting on top of C, so we have to have that in…
So, one of the actions is to put the log B on top of log C, right? So, in order for us to implement that kind of action, we have
precondition.
So, we have, as per condition, we have on.
B, comma X.
End.
Clear.
B.
And?
Smoke.
Beautiful.
So to put a B on top of something, right, the block B, right, for example, it should not have something else on top of it.
Right? It has to be clear, right? And, it has to sit
let's say if X is a table, it has to sit on top of the table.
So that is basically the condition. But we continue.
But, for… no, so we are writing… so this is part of the precondition.
We are adding now, elements of these preconditions. For example, this is basically as far as the Block B is concerned.
In order to move
the block B on top of the C, the block C, which is the Y here, has to be a block, so I'm writing block Y. So you understand, I'm not really writing the specific object names that I have in my problem statement here, right? I have… I'm defining now domain stuff, right? So I have to be general.
for assignments that the problem will do. So, assignment, to give you the end story so you don't get confused, I will put here what potential assignments the problem will specify.
The B will be the capital B block.
The X will be the table.
And the Y could be another block.
This instantiation of this Will actually be, part of the problem statement.
Here, we are very gen-genetic, okay?
So, we have here… another condition.
that… B and X are not the same type, And also, dot X…
is not the same type as the Y, and also, that B.
It's not, it's different than Y.
Okay.
So, another center is different, right?
it's a distinct, if you like, something grounded, right? Something that is grounded in the… if you can see, imagine it like a scene, something which is grounded in the scene is, are different. All the stuff that I involve in this argument are discrete things.
Okay, so that's basically the whole thing here. This conjunction is a precondition to be satisfied for the action to be taken.
And the effect here… Of this action schema.
is… on…
B, comma, Y, And… Clear.
X.
And… not… on B comma X.
And?
not.
Clear.
Why?
So when the big block, for example, was sitting, right, this location where it was sitting, the table, right, is now clear, right? And,
the block went on top of the other block, the Y block, right? The B block went on top of the Y block, and
The Y block is not clear anymore, which has something on top of it.
That's it. And of course, the…
The block is now on top of the table.
Because it now went into… to be on top of the block C.
Let's say, right? So within the block, C is still sitting on the table, right? So how can we have here it's…
No, no, we are saying the effect of the action.
That's the effect after you move the block B on top of the C. So, is C not on the table?
C is on the table, yes. So, so how can we have clear eggs?
No, X… X is… yeah, so, okay, this clear X has to do with, the location.
with a block B sitting, right?
But now it's clear.
So X is the location on the table where B was. X is…
Yes, X is the table. I mean, okay, now the table is obviously… Yeah, that's right. Yeah, so the…
The table is associated with a specific, sort of,
area where the block B is, in this interpretation, at least.
This is… This is the result of an action schema. So let me,
Let me show you some, booking a company.
other things to mention. Let me show you some examples of,
Actual problems, and domains.
So this kind of, this kind of block diagram kind of shows the…
planner, which is basically the solver that I was referring to earlier, that accepts these two files and results into a specific set of actions.
I have defined here what I just wrote, also.
And, some, examples of the
Of the, of the, the initial state and the closed state.
No.
Many people are working on, inserting.
into… that… The fine-tuning of language models, facts.
One way of doing it is with…
pure logic, like the knowledge base that we have said earlier, right? Now, if, however, you go beyond promotional logic, and you start thinking about how do I not mention the specific
instantiation of the objects, but to specific… but to mention rules, to express myself in some kind of rules, right? For example.
You know, that, blocks can be picked up only if they are clear, right?
the natural language that we express now, this kind of rules. People…
at, private research organizations. One of them is MIT.
I don't forget another professional name, but you can find the papers on the web. Converting… are converting the natural language into PDL expressions.
Keep your thought for a moment. Another group that is doing that is a researcher from NVIDIA called Chris Paxton, that he has also put some YouTube videos connecting BDDL and robotics.
And the reason, actually, there's a lot of interest in this type of domain-specific languages, and not only PDDL, but others, is that if you can express rules generically.
Then, you can… find a way that the model, the LLM, is…
They reduce the hallucinations because of the threats from these books.
That's basically the end result of this. So it's not, over here is, probably kind of a superficial treatment, on the constructs of the language, but, trust that it goes a little bit beyond that.
mistreatment.
So I would suggest also to take a closer look at this.
Verifiers and, how formal methods of verification can be exercise.
Especially when the, verification, requires,
Think about the action as a model response.
think about actions as model responses, right? And sometimes the model responses have to
Arrive in some kind of form of sequence to reason.
So, this chain of thoughts
that you're probably hearing in the reasoning kind of space. It's not entirely unrelated with what we're discussing here.
Because chain of thoughts can themselves be, Now, there's just be, type of, modulated by rules.
Alright, so, okay.
So this is an example… this is basically the file, right?
You can actually see that, we are defining the domain blocks world. We have the types, which is a block.
We have the specific predicates, own, clear, holding, hand empty, on table. This has to do with robotic arms, right? Hand empties robotic art.
And we have the action schemas.
Which have parameters, the type, and the precondition.
That I also defined over there a few moments ago.
and effects.
Okay, so we have two types of action schemas. One is for picking from the table, and the other is for picking from the block.
And also, all the actions that you can see. Put down, stack, etc, etc. It's a more elaborate example than my example.
So, this is basically a problem to the deal, an initial state and a closed state.
And here you can actually see the…
problem. We have a specific association of objects. I think there are 11 objects, if I remember correctly.
And, we have, some initial, state and some goal state.
That's all that is needed for a solver
To take these two files, Ben?
Start a search procedure.
I'm not 100% familiar with all the search algorithms that are happening inside this one. There is a kind of a…
The search algorithmic kind of space is quite vast.
And as a kind of a side note, the head of,
the visioning team of the Gemini team.
gave a very important lecture at Stanford, I think it was, like, a month ago or two months ago, very, very recently, right? And he referred to, very specifically, on the…
unsweetability of search for not… for reasoning in AI. So, I suggest that you watch this talk. I can send it to you in Discord if you are interested. So, the reasoning, direction that they are doing, is, not search-based.
Having said that, Search algorithms, maybe, Used as tools.
So the PDDL solver there could actually be a tool to, again, as I said, ground certain rules to reduce hallucinations.
But the core engineering elements of that is not research-based. Anyway, there is a VS Code plugin and a Python kind of library.
This service called plugin, if you install it, and you feed it
In one tab, the program PDDL and the domain PDDL files.
Copy and paste on that. The VSC plugin is calling the solver.
And this server now recently has become an MCP server, if I'm not mistaken, and returns back the solution.
You will see the solution, the sequence of actions.
And, that's basically the most… the simplest form of PDDL you can exercise in our mind. People have extended PDDL to account for temporal problems, people that… these actions take some latency to be executed.
And therefore, if you're dealing with systems that are latency sort of dependent, the solution may be quite different.
And, as I said, in robotics, there is a…
PlanSys, which is a ROS RO2 planning kind of system, that it is, sort of works with PDDL, again, for environments that are non-stochastic, and they… there's a lot of kind of repetition, like in manufacturing.
Okay, so I'm involving some kind of, side information here about,
I would call it a ring.
Solve the solvers, which are called,
They're mostly work with forward search kind of algorithms.
Audio shared by Hortcut.
But anyway, and some kind of video.
Which, I included over here because it is,
It is, useful to see how the VSport machine is working.
There is a… Planning Library.
the so-called Unified Planning Library, that it is,
with Python allow you to express, again, whatever you can express with the DSL language, but in a Python world. And some people find it useful to work on… with Python to do, to express the domain and the… and the problem. And, the Python code is also exercising,
The solvers in order for the solution to be presented to you on the…
On the screen. So you can actually take a look on this page as well. Unfortunately, it is a kind of a programming-oriented, so I do not know how to best to present it here, and I advise you to go and look at this
And I have two probably non-previous examples after this page. The logistics planning example, Which is,
A little bit more real, and, and, solve the problem of,
Optimal actions in the logistics space, like the picture we saw in the very beginning.
And, the manufacturing kind of robot, which is yet another example of a problem with its own domain and, and operations.
So these are basically the… what I wanted to mention about BDDL. It's just a more… more… most likely an encyclopedic usage at this moment in time for us, for our class. Having said that,
I prefer to focus our attention here on, reinforcement learning, and, but without… with all the caveats that I mentioned, that, classical planning may… and rule-based programming may still be, exercised in, in,
Reinforcement learning, fine-tuning, either for verification purposes, or, or, for, hallucination reductions.
With language models.
Okay, I hope you got something out of it. Okay, so, so let's now start the… Thanks.
The The easy subject of reinforcement learning.
No.
Reinforcement learning is, a domain that it is
it's a very long path to understanding it fully, with all these kind of algorithms that typically exist, but the most difficult part is to understand Markov decision processes, right? So the Markov decision pro… the journey to reinforcement learning, I'm not sure if you've seen, before,
Movies like, indiana Jones.
Indiana Jones movies, this guy walks into this,
Game or something, and, tries to find some treasure.
But on the left and right, there are some bodies, right, of previous people who attempted to do the same thing. So this is basically a good analogy on reinforcement learning, okay?
There are a lot of bodies around the trajectory, for here to the final thing where you understand a lot of things, or you can invent new algorithms and things like that.
So you have to take it slow.
If you rush.
things. Most likely, you are not really understanding fully, the previous things, and it's almost like a domino, like in many other domains.
And so, out of this kind of book, I'm focusing on a series of lectures that David Silver gave
David Silver is with Dean Mind at the University of Oxford.
I'm going to compress it to two and a half lectures of everything, right? These lectures of 12 lectures that you gave. But, as I said, after my Rust presentation, take your time and go back and watch these lectures in order, after the course ends.
Okay, I'll try to convey as much as I can. It's a highly mathematical topic, so expect to write quite a lot of equations, a lot of expectations, a lot of stuff like that, okay? So I'm going to present it in that way, okay? And I'm going to adopt
the Richard Saturn's terminology, that is exactly what he did as well, so if you understand the terminology here and the annotation, you will be in a good shape to continue.
So I'm going to start here with Marco.
Come on.
I changed this thing.
I changed this thing to… Okay.
Smartful.
Okay, something is wrong… is, what's happening with these things?
You know, what is the setting to reduce this, oh, maybe it's here.
This one, okay.
Mark both?
Decision.
processes.
MDP.
So, in a Markov decision process, obviously you're familiar with the Markov assumption from HMM.
We'll end up defining a probabilistic graphical model again.
That now will involve
Signal that the environment is sending us for every action we take, for every work.
Let me go back to the site so I can show you the block diagram that will start the discussion.
If I can go there.
Yep.
By the way, this page here contains the links to his lectures, the 12 lectures, or 14, I can't remember now, from David Silva, right? This is the page where you have to…
debits, Richard Sharpness's book is obviously free of charge, so you can download it and,
on something as the lectures progress, okay?
There is another book by Google engineers who was authored several years ago, before the advent of LLMs, and this is a little bit…
under the headline, Deep Reinforcement Learning, before LLMs were… became a thing, okay? So it's more… more on that kind of domain, and which I also link over here. This book is also free of charge, available to you, via the O'Reilly Library.
This as far as the background is concerned.
Well, let's start. So, we have here the…
Remember, we drew a block diagram when we were looking at perception, where the environment was on the right side, and the agent was on the left side.
Here, we see this kind of interaction. An action, as always, is using a state transition.
From S to S prime, as we call it. Here, they call it ST to ST plus 1.
And at the same time, a reward signal is available to us. Okay, so I'm gonna draw this thing a little bit differently.
Because it would make a little bit more sense.
to understand the notation, so I'm starting with ST is equal to S, capital letters.
with a P-index is going to be random variable names.
The small letters will correspond to…
values that are taken by these random variables. This is important to distinguish the two in your mind.
You have, an action.
We are taking an action. This is basically the responsibility of an agent to do so.
And you are transitioning.
2.
A new state, let me write a state first.
ST plus 1.
Is it called S prime?
The prime is, translated here, another derivative is translated here to be next state.
And we have RT plus 1, This is required to art.
A reward is a scalar.
Number?
And, obviously, we need to specify
The way that this scalar number is determined.
At some point.
So, 80 plus 1… Takes an X action, leads itself to some other
Reward and state, and then the whole thing is, repeated.
So, R is a reward.
A is an action.
S is a state.
And this thing is… Next.
Agent?
Environment is doing that. Agent… Environment, and so on.
This… sequential.
Sort of interaction with the environment.
is, sometimes expressed with these fridgelets, A, sorry, S, A, R, comma, S prime, A prime.
R prime, comma, dot dot dot. This thing is going to be called Experience.
And… We have, some kind of variables to…
To define as well.
Apart from the first four, which are self-evident, Capital T, is…
The time where we terminate the interaction.
The interaction can be terminated, either because
We decided to do so, right? The agent can take the decision to terminate the interaction, or usually it happens when you reach a terminating state.
Or simply run out of time.
Because sometimes we have…
problems that involve deadlines, the so-called finite horizon problems. We will not be dealing with finite horizon problems in this class. These finite horizon problems are more difficult than infinite horizon problems that we will be dealing with.
An episode is the interaction from t is equal to 0 to t capital delta T minus 1.
And a trajectory is, this sequence of experiences over an episode.
And we'll see what the return is in an equation very soon, and what is the discount factor, okay? So, we're now going to define the MDP problem, which is this calligraphic N.
To be… 5 double.
Calligraphic S, Calligraphic P, Calligraphic R, calligraphic A, and gamma.
This calligraphic S is the set Both states.
Allegraphic P is… transition model.
Calligraphic R is… Arity Wartz?
In a set of, set of rewards?
And, set of actions.
And gamma will be called a discount factor.
Given that the environment is stochastic, we already knew that from the HMM band of days.
We are… need to express everything probabilistically. No surprises there. We need to define certain probability distributions, and these probability distributions, the functional form of these probability distributions, should be consistent
with the probabilistic graph model, remember the whole discussion about the dynamic bias network when we did the localization, right? So we had as…
together with that, we need to define a convention as well.
The conversion back then is, first we take an action, and then we measure, right? We sense. That was the conversion back then that we adopted to specify these equations.
Here, I'm going to draw the probabilistic graphical model.
That now involves.
these random variables.
Now, someone could be saying, hold on a second, you told us the random variables are capital letters, okay, I don't want to delete everything now, and just…
replace it with capital letters in your notes, okay? Just the action node, the reward node, and the…
the state node, the nodes S and S prime. This is enough. Obviously, the whole thing continues over time, right? But I'm just here created just one instance that I would like to write now the convention.
The confession is…
that R and S prime.
R and S prime.
are determined.
At the same time.
Where do we execute?
the step function.
Joe, explain what the step function is?
The transition, in other words, right?
with arguments.
S comma A.
Are we, are we happy with the probabilistic graphical model? Okay, I think it makes some sense, but, the model effectively is consistent with the notion that the action is forging a state transition.
And, upon that kind of transition.
The reward and the next state are… going to be…
determined. Okay, obviously, the next state may not be the desired state that we were intending to go, because the environment is stochastic.
So we intended to go forward again, and but we ended up somewhere else.
So, we need now to define, this, kind of a loop.
Or…
This is just basically pseudochrome.
That, kind of characterizes everything we have kind of discussed.
We are, going to be,
Experiencing the environment, right, and all these interactions over a number of episodes.
We're gonna start with some initial state.
And, for the duration of each, Episode.
We'll take an action.
this step?
Function will give us a next state and the reward.
Okay?
You see the line, right? Okay, and
Then, we are going to do… Run the update.
And, if the interaction terminates for some reason, any reason, then,
when we go back to form another experience with the environment, right? Another episode.
So, now that we saw exactly what is going to happen.
We are describing these interactions with
an equation, a description called MTP dynamics.
Dynamics is a general word in robotics, but here we refer to MDP dynamics.
Which, I'm clear.
one specific, the MDB dynamics is going to be a probability distribution, and this is where the fun starts. This probability distribution will lead to definitions of the transition model that we met in HMM days.
And also, the reward.
modeled.
That, we are now defining.
Okay, so the MDB… this, probability distribution is going to be… The fee?
off.
S prime, R, given S comma K.
This probability distribution, and the diagram above it
R should be consistent. Do you see the consistency?
Huh?
These guys are determined.
Chrome.
dependencies that are from an S and the function.
This is given after the given symbol, you know.
So we'll take this, MDB dynamics, and with certain marginalizations, we'll define the two
models that we are going to be concerned with. The first is called the transition model.
Eventually, Morgan is gonna pee.
of S prime, given S comma A, Is equal to summation?
over R.
of P of S prime.
comma R, given, S, comma A.
Does this equation make any sense?
Yep.
Yes or no? Yes, it makes any sense. It is the marginalization operation, or the sum rule.
Right?
the… 2… Morgan?
is,
rewards.
model.
It gives us the probable distribution of a reward.
Given S comma A.
us.
A summation?
over all S prime states, P of S prime, comma R given S comma A. Yes.
I hope also this equation makes some sense.
For these equations, are, you'll also tell, tell us a kind of,
story. First of all, listening works.
Are going to be random.
Because of what?
Because we…
the next state, right, that we arrived, right, is not something which is, deterministic, right? So, there is, some stochasticity that is involved here,
And, sometimes the… The probability distribution of rewards He's,
how to refer to also a complete trajectory of experiences, right? So we'll see that how it enters
The objective function, because that's where we are going to define an objective function, that's where we are adding this equation.
Let's see, the transition model first.
The transition model is going to be kind of described with, Using a…
A rudimentary kind of world.
Concier still… Who are ourselves.
And, in this world, we have, the ability Too.
Move to a desired state.
With probability 80%.
But also, with probability 10%, We can move into states which are on the side.
from this world.
This world has two terminating states.
plus one.
minus 1.
And the job of the agent, Please, to start.
From the… an initial kind of state.
And, move in this environment.
In such a way that it will actually pick the… over the duration of an experience, that
Highest possible reward.
Now… We have gone and we defined, scalar
in all states with respect to this reward. So, for example, all cells…
Ponsense. I don't want to plug everything here.
gives us, when we reach this kind of S prime state into these cells, gives us a very slightly negative reward.
Why do you think this is it?
Why do you think this could be a good choice?
Because if we've had anything positive, Even 0.0001s.
this guy, We'll go around.
And around and around, constantly increasing the rewards forever.
So, obviously, we need to motivate to not motivate such a behavior. Now, this example is kind of trivial.
But if you'll, if you see what is also happening in the… Large models today,
Especially those who are optimized with the reinforcement learning, which most of the model is optimized with the reinforcement learning after the initial kind of next-gen prediction is done.
There are many places where some specific, parameter kind of setting will… unlock certain…
behavior from the LLM model, right? Which is not…
the one which was… ever could imagine it could result, right? So, every… many reinforcement learning kind of problems, agents have found trim solutions, and we had to go back and at least engineer them to avoid having this kind of trivial or
destructive kind of, behaviors, okay? So what is really the…
transition model over here. We are going to, have to specify a table.
Obviously, we have, states here as from S11, S12, S13.
That's the one.
S or 3.
That's, like, the direction of S prime.
And again, we have, the same thing as 1-1.
all the way to S.
43?
Okay.
And, for each, of the rows.
Let's take the first one, for example, just to go… And see what's happening.
Okay, so we have, if we move, up.
If we take the action to go up, Right?
We move to state S12,
With a probability 0.8, correct?
But also, we move to state S11. In fact, we stay to the state S11, because we can go sideways, we go against the wall, and we stay in the same state. That's the rule of the…
I don't think.
Obviously, there's a whole bunch of zeros here, but definitely one Other non-zero probabilities here.
And this table corresponds to… A is equal to A.
How many tables we should have?
Obviously, we have 4 tables, because we can take four possible actions. We can go up, we can go down, left and right, right? So, we can have…
as many tables, Us?
The action that is in my action set.
So in other words, this is the…
P of S prime, comma S comma A.
specification.
You can imagine also as a tensor, it's a three-dimensional tensor, that has this kind of probabilities for all possible actions that you can have.
But I wrote it as, kind of, four different tables.
the reward function I want to define now.
And also.
I'm actually gradually going towards defining objective functions, right? And I'm… in order for me to define objective functions, I have to define other things first, right? That's why there is a line at the end of a tunnel, in other words, right?
So, what is a reward function? Sometimes the reward functions are either… has two parameters or three parameters.
R, office comma 8.
The two-parameter version.
Yes.
An expectation?
of the random variable RT given
This is translated to English.
expected.
reward.
Received.
After.
we execute.
Action A.
from status.
Let's replace the expectation to approximate it with summations.
And… We are going to have.
Summation.
over R.
of summation.
of S prime, over S prime of P, of S prime, comma R, given S comma A.
Initially, this, definition Of this kind of reward function.
Is, a little bit cryptic until you realize that it is a very definition of an expectation function.
Because this guy over here, what is the definition of expectational function?
Expectation of X.
is what? The integral, I'll write the summation here, because it's convenient. X times P of X, correct?
So… Here I have R times P of R.
This thing is… field of art.
Given Escomai, correct?
So it's R times Q1. That's it, consistent with what I know about expectations.
And there's another, version.
That includes,
Now, this second kind of version means that the reward is dependent
also on where I will end up.
Right, so I have an S prime as an argument here. Therefore, I will expect to see this argument after the given symbol. ST-1 is equal to S.
80 minus 1 is equal to A,
ST is equal to S prime.
anybody… Replace the expectation to approximate with the summation.
So, I can see exactly the same, summation over R over here.
But because I have this… dependency on S pine of prime, I can write this as B.
Off.
S prime comma R, given S comma A.
divided by P of S prime, given S comma A.
Do you… do you see the… what I did here?
I'm conditioning into S with S prime now, right? So I take the joint, divided by…
what I want condition with.
So, P.
of X given Y is equal to 3 of X comma Y divided by P, Of Hawaii.
Right.
That's basically what I did as a parenthesis here.
Here, I have conveniently the MDB dynamics, which I told you is going to be everywhere, right? And also.
I have something that I defined earlier.
Which is, if you go up, you will see
The… this probability distribution is by transition model.
So, I am going now to write what is really the goal of this kind of agent.
Is to maximize
The total?
Amount.
Oh, reward.
It receives.
And I'm going to define.
Quantity that I will call return.
And I'll use a symbol Capital GT.
is equal to RT plus 1, That's comma.
RT plus 2.
That's gamma squared RT plus 3.
Flash dot dot dot.
is equal to summation.
from K is equal to 0.
to infinity, gamma to the power of k.
R?
P plus 1 plus K.
This, of course, characterizes infinite… Horizon.
problems.
Don't be confused with this infinite. Infinite does not mean that the Interaction will never finish.
Infinite means the fact that no one is imposing to me any deadline.
Again, look at your assignment form.
how… imposing deadlines, are changing.
The policy or the strategy of an agent.
In the last minute of every game.
Do you… do the players follow the same strategy as they were following in the first minute of the game? Probably not, right? So whether you have deadline, and, you know, obviously you're going to change the behavior, and these, problems are… turns out to be more difficult than not having deadlines.
So, because we have less constraints there.
So, we will focus on problems without deadlines.
So let me write it down. This is infant horizon problems.
In other words, no.
Deadline.
is imposed.
for a solution.
I hope everything is kind of, going kind of smoothly. I know there is a lot of definitions.
But at the same time, the provider distribution is something we have seen before.
To some extent.
Now, obviously, in this kind of return, that is, and this count factor here.
And this kind of discount factor can go from 0 to Very, very close to 1.
I justify you.
And, if it is exactly zero, What is the return?
Arctic Plus One.
If it is close to 1, they returned.
is such… that the agent is incentivized
for longer.
Corona.
I noticed.
In other words.
When a gamma… so gamma has to do with the net present value of money. It's better to have $1 million today than have $1 million next year.
Discounting is actually happening here for a very important kind of reason.
future rewards, Are discounted, because obviously they are more and more uncertain.
So I'm discounting now… I mean, they made a reward, I'm not discounting it, I'm taking the Arctic one, because I ended the S, kind of prime. But now that I ended this kind of Sprime, I may follow whatever trajectory.
After comes after that, because I'm operating in a stochastic kind of world.
And of course.
the environment can send me, any number. I can collect any number of rewards after that, status parameter in the time. So I'm discounting this kind of future rewards.
Appropriately, with this kind of a gamma, factor.
If the gamma is something which is actually very close to 1, I'm incentivizing the agent to consider only in the returns to be short-term returns. If the gamma is very close to 1,
sorry, if the gamma is very close to zero, I'm incentivizing the engine to consider short-term returns if, I am, have a gamma which is close to 1 long-term. Okay, that's the effect of gamma. And the gamma,
Your textbook has a very good example,
on, presenting the best strategy of an agent in this specific small world we just did the transitional model on, for various values of gamma, right? And, when the time comes, I'll show you the figure, how the…
And various types of rewards as well.
Okay, so… And if the opposite, gamma is close to zero, Did Daniel.
We incentive, sorry, we incentivize.
Both guns.
the policy.
It's the last definition we need to see before we start writing the objective functions.
The policy was symbolized by Pi, is a mapping.
From state?
to action.
And, when I say this kind of mapping.
Now, I want you to think about the policy as a probability distribution.
I will define this policy, I will denote that policy always as pi, of A given S,
Replace the pi with P.
to…
sort of, in your mind, notation-wise is going to remain pi. You always associate a policy with a probability distribution.
Dot.
It's going, obviously, involve manual elements.
the probability of taking an action A0, given that we are in the state, another probability… of,
Taking another action.
Given that we are in the same state, and so on and so on.
So… We call these kind of policies stochastic policies.
And obviously, we have a subset of them, a special case of them is the deterministic vortices.
Which, take a specific action with probability 1 every time you are in the state, you take the specific action.
Right.
So… deterministic.
What is she?
Yes.
Not pleased about the results.
from setting.
the probability of A given S is equal to 1.
A, bean, whatever.
Let me write it a bit clearly, because it's kind of important. P of AI given S is equal to 1. Some… some…
action i is always happening with probability 1.
That's a deterministic approach.
We've obtained our water quality is. This is basically our target variable.
We are going to,
discuss a little bit value functions. This is basically our objective functions.
Objective.
Value.
functions.
First, we will define them.
So there are two… two… Types of value functions. The first one, his state… value. Function.
And why is it a function? It's a function of the state.
beef… Imagine this room to be divided into grids?
cells, in other words, like the green world which just showed.
And obviously, the value and the task is here to exit the room.
And, there is one door over there.
Definitely the state here.
We'll have a lower value than the state next will go.
So, for every state, For every grid, in other words, here, state being location in this kind of case.
We'll have to have a number that manifests its value.
In other words, value, think about like a utility.
Yes, let's load everything from there, Lauren.
The reward, again, the reward is…
So, I think what you're asking is, It's not really the reward.
But they returned, because… The value diagram here is directly associated with
the trajectory that I will follow.
In potentially the future, to exit the room.
Okay? Okay? So it's not really that I'm going to be sitting here forever, and I'm going to have a value, right? I'm acting in the environment, right? So, there is some kind of a probabilistic return that I will be getting by executing and by interacting with the environment.
I will terminate only when I exit the room, when I hit a terminating kind of state. So, the value has to have something to do with this return.
So, in fact, if you see the definition, that…
VP of S, as we will call it, VP Office.
Is equal the expected Why are you over-policifying?
GT.
given SD, is equal to S.
Which is… Per the previous definitions, Summation.
Okay, is equal to 0 to infinity.
Gamma to the power of K.
R. T… plus 1 plus K.
given ST is equal to S, Which is…
Expected value over the policy pi.
Oof.
RT plus 1 plus gamma, RT plus 2 plus gamma square RT plus 3 plus dot dot dot.
ST is equal to S.
I hope you're following.
And if I… Goofy's… If I move this…
take the gamma out, and I have here RT plus 2.
Las gamma. Our team… Lastly… Class of the code.
given ST is equal to S.
we can show…
I'll explain, this kind of statement. This is obviously… the GT plus 1.
This obviously did… in a parentheses, it is the…
future return, in the T plus 1 case, It was one time.
We can show that VPI coppres, is the expectation over buy.
of T plus 1, the reward we can get immediately, plus the values… of future states.
Given esteem is equal to us.
And this… equation.
is… called the Bellman
expectation.
equation.
Or we buy offers.
The one who was, that's…
You know, a very important figure in this space of optimal control.
And worked with, dynamic programming, obviously, and other type of algorithms,
To address stochastic sequential problems and the optimal kind of solutions.
Is responsible for this?
Equation, and then derivation of this equation, which is actually outside of the
what I wrote over here. The derivation for those who are interested in is inside the Bellman book, but I cannot write all the derivations here, otherwise we'll never finish.
This equation is important because it connects the value of future states to the value
or the states I came from.
In other words, it defines…
Some kind of, what we'll call buck up.
Three.
Where?
I'm starting, let's say, from a state.
I'm responsible for a set of actions. The solid lines in Rita Sato's book, as well, is… indicates actions.
So, from a state I can take any set of actions, Ms. Gordon?
development equation, shouldn't there be… Some dependence on discount.
Yes, Bill, there is a gun, there's a gun, yeah, so yeah, please. What is it?
There's a Gava here, right?
Yeah, yeah, there's a ground to do that.
So, as I was saying, I'm starting from, let's say, I'm now in state S, and from state S, I can take a set of actions.
The set of actions the set of actions that I can take.
The specific action I'll take is going to be
Finally, determined by the policy buyer.
Because that's what determines the action I can take. I can sample this probability distribution, and at some point, I will take an action out of it, right? The sample will correspond to an action, it's a discrete probability distribution, right?
So, I'm arriving at a solid node over here, let's assume that I took this out.
After… complying with the policy pie?
From that kind of action, what will gonna happen?
Remember the probabilistic graphical model? The action will induce what?
a state transition.
Who is responsible to tell me
To determine, if you like, the possible kind of states.
The transition from London.
So this is over here.
this edge here, I can write PSS prime.
How is your mother.
the same action.
Can lead to one state, or multiple states.
multiple states.
are possible by taking one action. Because, as I mentioned, the model is, Stochastic?
And therefore.
These next states… are associated with Fine.
themselves. Each one has a line.
Remember the value? In the case of the grid war that we are in right now, is, mathematics.
The value function is a matrix. Every state here has a byte.
Obviously.
we'll see how we'll estimate these values, but obviously the end result is that the value function is going to be a function of S. This S could be anything here, so it's a… it's a…
it's a… if I… if I have 50 squares by 20 squares, it is,
Has 200 elements this function.
So, what I was saying… I was saying something about this kind of backup tree. This equation connects.
These two qualities here.
Visible point is here.
And allows us the estimation In a… as it turns out, in an iterative way, or recursive way.
I want to mention something, important.
In this kind of expectation, Over here, you see something which is… always is notation. Expectation.
With respect to the policy, right?
So, I want to, clarify, a little bit this, this kind of notation.
So let's assume that I have a policy.
of A given S, which is a probability distribution, 0.25, 0.25, 0.25, and 0.25.
In other words, I have a uniform probability distribution across all my actions. I'm in this kind of square, and I have an equal probability to select any of my four actions that I have, right? That's what this pi of A given says.
So… I want to make sure that we understand that a stochastic policy
does not
make… the MTP stochastic.
But the stochasticity of the MDP is because of this transition model itself. So, in other words, I could express exactly the same equation, but instead of the expectation over pi, I will put expectation over S prime.
I want to show you that it's one and the same thing, because S prime is what?
is a next state that is determined by the MGB dynamics, correct?
By the transition model.
Which, if I may write it as S prime.
Given.
S comma A.
Right?
This one… this… this thing can be written Us.
expectation over… Sprague.
that from the distribution, S prime given S comma pi of A given S.
Huh?
From a heck of an ease, That thing over here is…
Because we are kind of,
kind of condition on it, right, for the specific, has a probability distribution. And we have… Obviously.
Determines what kind of action we'll see subsequently, right?
is giving, if you like, the… is responsible for this expectation operator. The expectation operator behind this kind of pi, there's an S prime.
Okay.
So… So let's see what is, the next…
And final value function we will introduce.
Predicted.
I have, I have sat down and derived
this equation. So, for the derivation.
In fact, I remember it's not fully in the,
Richard Saturn's book, strangely enough, but I characterized it for the derivation.
off.
the Bellman.
expectation.
equation.
I'm gonna send you, I'm gonna take a photo and post it on Discord, right? You can actually see
The derivation is, I don't know.
Okay, the measure is, like, two pages.
Two pages long.
But, I think it's worth, not necessarily right now, but at least worth in going over the derivation to understand how the…
Sort of, this recursion was established.
to discover the message.
I'll… I'll send it to… Either tonight or tomorrow.
So, we are going to now write, this Betterman expectation equation.
In a… in a way that will allow us to also estimate the value function.
So via kind of a trivial kind of problem. I have exactly the same world as I had earlier.
With 1, 2, 3, 4 states.
Thank you.
And I had this, transaction model.
So, the problem, is called the literature prediction problem.
We are doing prediction first, and then we are solving the problem, which is called the control problem, okay? So we are going… in the prediction problem, we are able to define the value
For states, for all the states in our problem, and as it turns out, it is
going to be an iterative process, but I'll show you how it is done.
And the prediction is the following. Evaluate.
The VIPAR office.
Given… a policy.
P.
of A given S.
Addiction is… The evaluation of a policy.
The control will change the policy.
Which, of course, leads us to…
Re-evaluated, and then take another control action to modify them more and more and more, and we will show that the optimal policy is obtained at the end of this iteration.
So, we're going to use Bellman optimality equations next time to define this optimization that drives the control algorithms, right?
And there are two control members who will see the value duration and the policy duration.
And this algorithm will actually lead us to defining the pi star, the optimal policy for the specific problem, assuming
the transition model, and a certain reward function that we will need to define. So these are basically what we'll do. Now, I don't want to start it because
is going to take a while to write down all this kind of evaluations, and then I don't want to interrupt it in 5 minutes or 4 minutes. So I'm going to stop here, because I think you had also enough.
From the discussion here tonight, today. Okay, so I will continue next time.
What?
If you sign in, everyone sign in? Who has that?
But, yeah, I don't think that…