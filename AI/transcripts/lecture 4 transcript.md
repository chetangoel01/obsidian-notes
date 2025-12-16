Okay, let's get started. Please be seated, and don't forget to sign in during the break.
Which we may or may not have, okay? All right, so, let us,
Let us try to sync what we have done in the past.
Apparently, one week is a long time with everything that's going on. Okay, what is… Okay.
Last week, we,
Lastly, finish, buying the classification.
set up professional audio and holding acceptance.
Alright,
And, what is really going on with this kind of, classifiers? We show some, kind of metrics.
the report and the precision, and we also organized the fact that we can definitely report some trade-offs between false posit and false negatives, sorry, between true post and false positives via the ROC curve. Okay, so remember the ROC curve that we looked, last week?
By the way, I forgot to mention the one logistical kind of question I had in the beginning. Your assignment deadline is when? Sunday. Sunday, okay? So, I saw a lot of tickets, right? I see the TAs are responding, but I also, also got the statistics from attendance of the TA office hours, right? From 0, we went to, 5.
We are kind of 60 people here, so maybe you should put some more emphasis on going to the… to ask this question right on the spot, rather than opening tickets. We see a lot of tickets with environment stuff, and so I'm not so sure if they will be able to respond to all the tickets by the deadline.
Okay? And that's why I asked you to start the deadline, sorry, the preparation for that assignment sooner, okay? So that's, lessons there for now, for next week, for next, for next assignment. Assignment 2 is going to be published this weekend. It will be due, when is the middle.
17th. The 17th, okay. So it'll be due, one week before the 17th.
One week before, so it will be, probably, yeah, I'll need to see. I don't want to coincide with them either, right? So I need to see exactly what they assign will be.
If we don't have time. Time flash, already in the mid… we are kind of going into October, I think. Okay, so let's see, I was just showing you…
the binary cross entropy. Can someone tell me, what is the highlight of this kind of binary cross-entropy? What was really discussed, back then? I mean, we actually…
started with a maximum general formula, and you plugged in what type of P model.
And obviously, when we have multi-class specification, what we will plug in?
Okay, so for whatever is appropriate for the specific task, we plug in the corresponding probability distribution, and, chances are, even if
in the general sense, we are not able to manipulate explicit derivatives, like for that, for gradients. Chances are that your framework that we're working with has the numerical derivative condition.
So you can actually do, as we did with, backward migration kind of procedure, a bit later, the… solve the problem, the optimization problem, during…
Okay, so we don't have a… well, we didn't demonstrate an American kind of part there, but it is obviously, of course, probably would have a chance to demonstrate it.
So we saw something, like this, and we basically boxed this kind of main conclusion that this, loss is, heavily penalizing confident wrong decisions.
Because of the asymmetry that we see in respect to the Y-hat. What is the Y hat? Represent always?
Prediction. It's a prediction by what represents probability.
The probability of… of 1, the positive. The probability of the first plus. Right now, this is what it is, and in the future.
After we go through, you know, we show the specific kind of classifier they call logistic progression, and we start putting together this kind of simple single model neurons, right? In what we call our first kind of neural network kind of implementation.
And the aim there was to do what? To introduce more complexity into the hypothesis, because we saw that this classifier, a similar neural one, was not able to classify this more complicated kind of dataset that we saw in the TensorFlow playground.
And, the main conclusion of that discussion was that the network is building features, while at the same time is working to make better decisions. So everything is jointly optimized. Both features are… imagine all features being a point in a…
N-dimensional space, okay, whatever dimensional space your network kind of duplicates.
And as the training goes, the features are actually moving, because they are changing. So that's the visualization we are… we are able to see, and of course.
this means that the joint optimizer will find a theta star, and where that will do the inference. And if the probability distribution that we have, during inference is a little bit further away from the probability distribution that we use to train the thing, obviously we'll see certain
bad behavior, right? So there's always this kind of concept, oh, I didn't explain it, of drift, right? Of what's called model drift.
a fully connected, network. So, we said that we are going to, obviously, be able… should… we should be able to put,
many neurons to work for us in a layer, as we call it, and also we have multiple layers that are stuck on top of each other to form this kind of a pyramidal kind of shape, where at the bottom is my axis, the raw data, and at the top is my prediction. Whatever prediction that is.
And, we saw that,
The equation for that is, this time. H is relative of WX plus B. That's the equation of the dense layer. We have, shown, we have proven that in a trivial kind of example, yes, indeed, this layer contains two neurons.
which is also my dimensionality of my age, of my feature vector, in other words. That represents what? What always does the feature vector represent?
represents VX.
That's what is a representation of X. We'll call this a representation of X. Now, obviously, it's a… every layer is doing its own kind of thing, so it's… it basically projects the input, its own kind of input, but ultimately, all of them are building a feature at the top of this network.
This right there, just before the head, if that is dimensionality, let's say, n prime, we have an N prime dimensional vector over here, which represents X, okay? And over here represents
If it is a classification problem, the specific class, right? So when you see a cut during kind of inference.
the age vector will be the picture that we extracted from the concept of cut.
Okay, so we represent the count.
Nope.
Okay, so finally, we, built a… we used this kind of, dense kind of layers to build an understanding about what is going to be the Y-cat when you have multi-class specification, and we presented the softmax.
What was the role of the numerator in the short blocks?
what Rome does in America thinks?
I see that the other one is studying hard during the previous lectures, material. Anyone watch the video at all? No?
grab a cake.
I have one glass one in here that, The audio was not…
There, okay? And everyone discovered that in the weekend before the leader.
Very little things can be done at that point. So just make sure that the O2 is there, just double check. Okay, so what is this… this numerator was doing what? So look at the, the question that I can ask before going there, and then it will be evident what the role is playing. Can any element of Z be negative?
Good.
No. The answer is yes, it can, because, obviously the value, the H1, produces either zero or positive elements, right? But look what happens after that.
We have a W matrix by vector multiplication, and but we also have a bias addition, and the bias would be a negative number.
And therefore, you can take any positive number that comes over here, and you can add a larger negative number to it, and therefore, the elements of Z will be negative, right? And obviously, we need to make them.
Positive equal to the power of negative bits, positive equal to the power of positive.
So we have, the… taking care of the… of that kind of, thing in the… with the numerator, and, obviously, we need to satisfy the other constraint, this constraint over there with the denominator.
Okay, yes, no?
Good, okay.
Finally, we are getting our posterior probability distribution now, distribution, that, we are definitely going to report this specific class that won the competition.
But at the same time, we are going to…
report the confidence here. And I… I think I asked a question maybe then, I'm not sure, but evidence in the confidence we have for this class will be affected by these other classes, or no?
Yes, it will be affected. Some people say no, some people say yes. Yes, it will be affected, because of the nature of that softmax equation.
In other words, if I had another confident, almost as confident as the winning class next week, right.
And you do the calculation, you will see that the confidence of
the class that finally won the competition is actually much smaller than if you have a situation such as this, a nice situation such as this. In other words, I have to distribute
weights or confidences around, right? So if I occupy a competing class last confidence, this means that my confidence also went straight, right? The winning class. So the… so you may win the competition, but, you know, your confidence may…
be, dependent on, what others are we?
Thank you.
Okay, finally, so they'll… so we introduced the,
multi-class classification, the binary percentropy, or the multi-class percentage is there as a loss function.
And, we are, basically, also in other observations, we saw that these kind of test networks are very, very expensive, okay? We went into… in that example, in your textbook, Python textbook, that is, we went, I think, to 266,000 parameters, right? And,
Someone has to do the grading calculation
For 266,000 parameters, what is the length of the gradient vector?
What is the length of the gradient vector for 266,000 parameters?
No one.
266,000.
Because the gradient vector is the derivative, each element, the derivative of the loss with respect to parameter 1, parameter 2, parameter 3, parameter 266,000.
We have done this… I asked this question before, when we're doing even linear regression, the partial derivative of that mean squared error with respect to the hypothesis. Back then, we had 2, 9, whatever, some small number of parameters. That was the size of the grading factor.
Okay.
All right, so I basically suggested that this kind of black propagation over here will be, done from a kind of first principles.
In a sense that it's a deterministic procedure.
that existed many years back, but over here it's going to be explained, and the procedure nowadays is obviously exercised every time we train a very large-scale neural network. And of course, we have many, many complications when we want to
expand across multiple GPUs these days, and multiple servers within the GPU… within a cluster of compute, but that's another kind of story. And it involves a forward pass and a backwards pass. Okay, so what is the forward pass is doing?
They're simply doing what?
No one remembers anything, though. Okay? It is… it is gradually calculating the value of the function for a specific values of X and Y. Specific values of X and Y are coming in the input.
And it explains in an equation form, using primitive gates, as we call these individual kind of computational units, the value of the function f. Okay, that was the easy forward pass. What was the backward pass was doing?
was using the chain rule, as we said. We saw it in a template, fashion over here at the right-hand side. And at every step in the way.
starting from equation 8, going to 7, going to 6, and so on and so on, we are propagating its gate. So the chain rule was saying, I have received an upstream gradient, okay, plug that in, that is your DZ in that equation, and all I need to do is to do what? What?
is the only operation that has to be done to be… for each key that we meet, going from the top of the diagram to the bottom. The only operation here is this guy, the local gradient calculation, the local gradient.
So the local gradient is,
parcel delivery of the function of the gate, where do I pick up the functionality of the gate? From the forward equation, right? So we said that we are going to have an multiplication, that's what the gate is doing. At the very top of the tag, if you go back to your notes.
And, we are going to, obviously, in the… do the partial delivery with respect to the port.
to the port that I'm doing the specific downstream data calculation. And then I'm looking at some primitive lookup tables. Did you print these lookup tables? Do you have them available in a paper form? Because October 17, you're going to come here, you need this paper.
Okay, you need this paper and a physical paper. No one's going to be able to use anything other than a physical paper, at that time. So you need your physical paper notes, and you also need your, your, some of the additional things that I'm mentioning. So where do you find this, this page?
What do you find this partial derivative lookup table?
I mentioned that. It is in the website called, math, resor… resources, math.
And then there is a calculus section on the website, and in there, there is a lookup table. Okay, let's see.
No peso horses.
That's your course, okay, fine.
Okay, so you go to the website, and here at the bottom, you see calculus, right? Math background calculus, okay?
So, this is Gordon.
Table of simple derivatives.
True.
Table of simple delivery. Some… just in case someone asks you a kind of a trivial backup location question, you need to have that lookup table.
So, which entry have we… so if you have a multiplier, like the multiplier on equation 8… equation 8, which entry
Do we pick up from here? From this?
From this, I see a lot of equations here. Which one would you pick to consult? That's the… that's the thing that you need to be familiar with.
So we have, what was the expression? Inv the norm? What was that?
Partial derivative of nu, this is with respect to nu. This one cannot be replaced with X.
So, X times something… scalar in this case, right? With respect to X. And therefore, you will pick up…
This guy over here.
All right? There's no X here, but there is a function f, right? But if f is equal to X, you have C times partial derivative of X with respect to X, which is 1, therefore the answer is C, therefore that's why we wrote
That answer in the…
response here. I saw that this is the answer. This is the scene in the lookup table. So you have to have this kind of mapping between the tensor and the, sort of, elementary school,
look updated.
Alright, so the same thing happens with the other port of the multiplication kind of gate. It's exactly symmetrical problem, but now the C is swapped.
Right? And, you have none.
And then, you finally calculate the… you go to equation 7, you go from 8 to 7, you do exactly the same thing. You will be able to find, yet another time, another entry.
in the, over here, which entry would you be able to use? Which entry? This guy. Okay. With this guy, so it is,
minus 1 over X squared, okay?
value in some…
textbooks of calculus, the prime is a derivative notation. You probably can know that from hidden card. Okay, so the partial… so this is basically the 1 over the num squared, known times 1 over the num squared. Mind you, every step of the way.
I… this is a number. There is a number behind it. I have it in the memory. Where is known was calculated when I was doing forward propagation, I store these numbers in memory, right? So I'm able to retrieve them and use them
Right now, when I see this delta, I'm just plugging them in. I see… I also calculated this guy at some point in the forward pass, so this tensor now has a value, a specific numerical value, it's not an equation per se, right? It's just numbers. I just do not have numerics here, so I didn't write them.
Okay, so what was the conclusion that we can draw from transfer variation of the two gates? And I asked you, I think, I asked you to go back home and do the buffer ligation for the remaining gates, at least two or three, just to get the gist out of it. So, who has done it?
Great, okay. You, you will do great in the meantime. Okay, so, this, so, so what was the main…
Region of acceleration here. Parallelism. Okay, that's the main factor of parallelism. Where is this parallelism here in this combinational graph?
How did this parallelism manifested? Yes, when the people…
There are independent branches. Both of these branches will receive some kind of upstream grader from someone, and therefore local gates can be parallelized. Execution of these downstream graders can be parallelized.
Okay, what was the second factor?
The second factor was that we have elemental ATAs.
And this elemental engaged means that we have a lookup table. If we didn't have elemental engaged.
who only happens to be able to symbolically look up, so we can expand it from a squaring function over here to something that is matux by vector multiplication given.
We can go do the calculation once, store it symbolically in a lookout table, and every time we have a matrix by vector multiplication, we pick up the
value that, the symbolic result from the lookup table. We plug in the previous calculated kind of tensors, we are done.
And the third thing was, obviously.
The fact that we were able to
We use quite a lot, there in the results. That's kind of self-evical from the procedure.
Any questions to that, trivial example.
Okay, do you want to do one more example from a kind of a single neuron kind of a thing? Okay, so let's see, I think I have an example over here.
for the background migration of…
This is what? This is, lecture.
Is it at 5 today? 4. 9?
So I have this kind of a trivial, sigmoidal urine over here, and I'm giving you some values for W.
Obviously, this is W0, W1, W2. I want to use this general W in the calculations. And also, I'm giving you the specific input X.
which is, X0, X1, X2 is equal to minus 1, minus 2, and 1.
Okay.
Alright, and what is the question? The question is…
write the question here. The partial derivative.
of Y hat, so the gradient of Y hat with respect to W. Of course, we may have a loss, a binary cosinography loss, and obviously we will calculate the gradient of the loss with respect to W, but I want to make this example a little bit easier, so I'm just treating a specific…
location in the more elaborate kind of example someone could draw here.
So, obviously, the Y hat, this gradient will have, how many parameters, how many entries the gradient will have.
How many?
It's evident, because the number of parameters here is a vector of three values, right? So, it is a partial derivative of Y hat with respect to W0,
Okay.
of Y hat with respect to W1, and partiality of Y hat with respect to W2. These are my entries of my, of my gradient vector, 3 entries.
Okay, so if I go to the forward bus.
I have, I need to… I need to forward propagate
In other words, I'm going to start from the point, as we said, of least debate and C, right? And at the end of the day, I'm going to have the calculation of Y hat. So, anyone has any ideas, what is the first equation?
Z is equal to…
W transpose X, right? Again, we said that we're gonna do gradual calculation. Okay, that's equation number one.
All right, so… and evidently, as we are doing the forum propagation, we need to obtain numerical values that we have to store somewhere, right? So, can we calculate Z? Obviously, we calculate Z. If you do the dot product, what you will get?
We'll get to 1.0.
Okay, that's the number you'll get with Z.
A scalar, obviously.
What is your second equation?
What is the second equation? Y hat equals sigma… Y hat is sigma of Z. That's the second equation.
And if you do the calculation, it's, 0.73.
Okay, are we following what's going on? Okay, that's a forward pass, that's an easy… that's an easy part. Okay, so let's write down the backward pass.
Eliminated, over here, so I have this.
Okay.
Remember the template. Your template must be somewhere in your notes, right? Okay, so, so what… which… where do I start? I start from, or the end, the last equation. That's equation number two, and I want to back over my data.
So, however, to back up my data question number 2, I need a kick-starting gradient, right, which is the partial derivative of Y hat with respect to Y hat. That is 1.
Okay, and now I'm able to apply the template.
which, I have now a gate that involves the SA sigmoid, and that sigmoid has only one input port. Therefore, I have
One… downstream gradient that I can write, I can name it.
I will name it, and I'm not here.
And DZ.
What is the equation? The downstream gradient is equal to the abstin gradient, which is DY hat.
partial derivative of the gate. What is the function of the gate?
sigma of Z, let's say the tensor is called Z, with respect to Z. Okay. If you go to a lookup table, in, sort of, from that we saw, like, a few moments ago, you won't find this, partial derivative there, and,
Sorry, if someone is giving you a problem like this, then we give you to give us a formula, right? So the partial derivative of the sigmoidal unit.
is, first of all, this guy is 1, it's sigma of Z, 1 minus sigma of Z. That's a form of it. That's a symbolic…
You pick it up from a lookup table, hypothetical lookup table, and you write down the equation. Do you have a number there or not?
Yes, because I know Z, therefore I have a number. If you do the numbers.
The result will be 0.2.
Alright, so I hope you recognize I'm not really doing anything intelligent here, I'm just following blindly the procedure.
So what is… where do we go next?
What do we go next?
Equation 1. Okay, I'm done with equation 2, I'm doing equation 1. Now…
At every point, you have to also be aware of the question, right?
Because if you go and write down, these.
What is the mistake we have done?
Just made the mistake.
It's not a catastrophic mistake, but it's a mistake.
We are… We are actually going and, calculating downstream gradients that no one is asking us about.
So no one asks you anything about DX. The only thing that they ask you is the DW.
Right? So don't… make sure that in an exam setting, you don't go and calculate whatever, right? But you be careful what… what the question is asking. Okay, so… so the DW, so that's the only thing we need. DW is equal PZ, right?
With respect to…
Sorry, and the local gradient is the function of the gate, or at least a function of the gate is a not product.
with respect to… With respect to what?
With respect to what?
everybody would respect room?
to the port, right? Which is… We're calculating the…
Partial derivative of… what is this? W. W.
W is a vector.
We moved a little bit away from this scalar world that we were in the previous kind of exercise, and now, we have to…
sort of know a little bit about, sort of, gradients, and I think I asked you to…
also review some of the gradients in Khan Academy, right? And there were also some kind of formulas also on your site. I'll show you where they are.
But now we have a partial derivative of a scalar with respect
to a vector, okay? So, let me write it down. Do we know DC? Yes, we do. That is 0.2.
The partial derivative of a scalar with respect to the vector is… I'm going to write Zone.
The elements of this… of this expression over here.
Okay, so it is positive derivative.
of W0.
X0 plus W1X1 plus W2X2.
Do you agree?
With respect to what?
WCU. WCU.
Then we have partial derivative of exactly the same thing, I'm not going to write it with respect to W1.
And then we have partial derivative of exactly the same thing with respect to W,
2. Okay, so this is my… these are my three element…
vector now that I need to calculate, okay, the partial derivatives.
So…
I'm going to the lookup table again, and definitely I will see something I already have seen before, even in the trivial previous example.
So what's… what is the simplification here I can do? This… this summation is partial derivative of W0X0 with respect to W0, plus partial derivative of W1X1 with respect to W0,
Plus, that's the derivative of W2X2 with respect to W0. Do you agree? Okay, so… This thing is what?
Zero, this thing is also zero.
And the only thing that remains here
is elements such as this. What is this result to? If you go to the lookup table, it is results into X0.
Okay, so I'm going to do exactly the same manipulation over here, and finally, the end result will be 0.2.
X0, X1, X2.
Oh, this is, kind of an example that
Someone would say, okay, it's not really entirely kind of innocent, in the sense that it does show that certain
Partial derivatives and gradient flows inside, even a trivial kind of thing, depends on to the input.
Okay? And so, in many textbooks, you may see, even in your own kind of textbook, you may see, okay, if one of the features of my input dominates the others, what's going to happen?
the gradient will be pointing always towards the dominating feature direction.
Which means that,
Even if I have a stochastic kind of thing, right, I will not be able to move around so much.
And if I'm not able to move around so much, is this bad news or good news?
But, man, bad news. We saw that.
We saw that in the… in what we discussed a bit earlier. I need to move around to escape from… from what? From local or global.
Now I'm regretting I gave this example because of an excellent question, Yomitan. Let's go. So, is it because, like, one of the features is very dominating? Yes. So, when we have a large dynamic range differences between features, what is the nice thing to do?
Normally, actually. So my question is actually, had we been using gradient descent instead of stochastic gradient descent, the gradient would still always point in the direction of, like, the… like, the minima, right? Even if it is not towards the other features. Even if the… even if one feature is dominating.
Yeah, so when you have the stochasticity, right, obviously you have other effects. You have also the kind of thing. That's basically where most of the… I mean, the noise that we're producing in the gradient. But if, in each minibatch, you have something which is very, I will call it,
systematic, right? You will not have as much, sort of, movement. So is it safe to say that, like, if we were using gradient descent, like, not the… not stochastic?
So, we could certainly get better results with our normal.
Not really, because, okay, in this trivial example, maybe. I do not know, okay? But in general, no. In general, yes.
Anyway, some interesting stuff. Okay, now I hope you got the gist of this thing. Now, you may see, maybe a bit shocked about this kind of analytical thing that is going on with respect to scalar, with respect to vector.
vector with respect to a vector, and so on and so on, so there are certain formulas in calculus, right? They could just…
treat them as lookup tables, okay? We don't need to know, although the derivations that I'll show you were the derivations, so you didn't know how they were derived. We can accept them as ground truth and just use them in any problem that is involving some more elaborate example that we may have here. Okay, so let me show you where they are, first of all.
So where is it?
Okay, so if we go to… Beautiful.
A bit more furnished and get, strained.
Okay, if you go to,
software vacation.
a propagation analysis.
So, do you see this kind of link over there?
this link, this text. This is one preference.
Okay, this is a computing neural network kind of gradients, okay? It actually goes
into calculating how the, I will call it, more elaborate lookup tables for cases that are typically met in neural networks were designed.
And so, if I have, for example, what is this?
Vector matrix multiplication with respect to vector, right?
This is what you store at your symbolic table.
you take the matrix, you transpose it, that's basically the end result. You don't have to do anything else, if you have something like that.
So there is a… there is some kind of hope that not everything is so… I would call it,
And I think last week, I think I showed you this last week. Well, you don't see anything.
If I delete.
Okay.
So this is basically an example of what we have typically, let's say, in a neural network. You have a cross-centropy loss, you have the softmax. Before the softmax, you had some kind of a vector mathematics multiplication, and
If someone is asking you to calculate the gradient of the loss with respect to a tensor, here in this case it's Z,
you just take out the, prediction vector, subtract it from each Y, the ground rules, and you're done.
You don't have to go in every step of the way, in other words, right? So you just use lookup tables. That's exactly what's happening inside the framework. Symbolic computation.
If we can, of course, do it. If you cannot do it, then obviously we'll have numerical ways of computing, with other difficulties, say.
Have a good evening.
So now, anyway, take a look at this.
exercises also, that I have in the site, just to be familiar a little bit with this site. No one is going to ask you to do something extremely complicated here, but it may be a question that,
Kind of, get you, 10 points or something.
Okay, good job.
Alright, so let's now move on a little bit.
So I'm going to, there are some other sections over here which I'm going to skip.
Because I want to go today into convolutional neural networks, and when we do convolutional networks, we'll go back into those kind of sections, because when we look at computer vision, we will, see, you know, a little bit larger kind of data sets, where this kind of things, like regularization and, I would call it,
batch normalization and things like that, these type of elements will be a bit more useful for us. More… not useful, but they're still useful in DAS networks, but
there will be a bit more evidence through the examples, you know, when you have a residual network, there will be batch normalization blocks in there, you know, all this type of stuff, right? So we'll understand them when we go into CNNs.
Okay, so this is what I want to do next. I want to move on into a little bit another application domain in the computer vision domain, and this is where we stay for a while.
In fact, a computer vision domain will be assumed all the way until… your, retail.
Okay, so in the meantime, you would expect to see some revolutionary network kind of questions, and whatever we'll manage to cover until the week before your midterm.
So let's do that.
Sorry if someone is asking you, let me see if we're having pictures here.
reception.
What's the first thing that you notice when you see a scene like this?
The first thing. The taxi. That's basically where everyone responds, first of all, by construction, you're focusing in the middle, right?
And, also by construction, from some, force many billion years from you.
also notice bright colors, right? Because we are constructed to either find food or
to run away from threats. So if the yellow tiger is coming towards you, you just run away. And so, usually.
fruits.
Tomatoes, whatever, are bright colors, right? So you…
Attract your attention, that there's no coincidence on cabs, except from…
one country. The caps have a yellow color. All right, so, in fact, when we look at this kind of scene, we do two things. One, with a very, very small latency, we process the food versus threat kind of situation, right?
And then what we do is we start sampling the scene, And…
sort of, almost kind of fine-tune our attention to the task that we have at hand. So, for example, when we were awaiting.
sort of, for a person to arrive, we, started focusing on people coming out of cars, you know, people are coming towards us, things like that, right? So, probably, you have your own experience about that.
Okay, so that's basically what I wanted to mention. So if I go…
and ask you the following kind of trivial question, just to start this kind of discussion. If someone is giving you, someone is giving you the following kind of problem.
And this is the introduction to… Little correlation.
on pollution.
I have a… Some kind of a memory here.
And inside this kind of memory, there is a…
some form of a signal over there. She's like, why are you?
Some kind of buffer.
And someone is asking the following question. Is there any way that you can find the location of this
thing which I have shaded. Is there any way that you guys think of it?
Any ideas? Finding its location. We don't know where it is. We're somewhere in this buffer.
But we do not know the agency.
Remember the radar example? That's a classic problem in the radar case. Also, when we receive the returns, right, from these planes, the returns will be somewhere in this kind of buffer, right? It will be obviously not very well
Sort of constructed, such as these square files over there, but it would be something there.
We use exactly the same concept.
So… I forgot to mention that someone is also telling you that you know the specific
Shape of what you're looking for.
So, one idea, okay, one idea to say, okay, I have,
I have some other kind of buffer over there, and since you gave me the specific kind of shape.
I'm going to locate that thing at the very beginning of the buffer, right? And I'm going to execute the following kind of operation.
I'm going to multiply.
X of T, NYT.
And, some… the result.
Okay, so what is the… if you do this operation for that specific location, what will be the result?
Zero.
Okay.
So, let's now do the following. Let me try another location where this thing is located. So, I'm sliding this to the right.
And I'm doing the same thing, and I'm doing the same thing, so obviously I'm getting another zero, I'm getting another zero, and so on and so on. At some point.
This thing will be somewhere here.
It will start to overlap.
with the X of T, right, contents, and I'm gonna start getting I'm gonna start getting… probably,
a slightly non-zero value, and maybe a larger value, and right when we have complete overlap, I'm going to get a pick.
And, very symmetrically, After that, this thing will die down.
So, right at the peak, I'm claiming is the location of the X of P, which makes kind of some sense, visually at least. And there is an associated kind of equation, which I'm going to show you in a moment, that is doing this now for the following kind of question.
What if I expand the dimensionality from one dimension.
like, in the time dimension, or the buffer dimension, to two dimensions, right? And in fact, I'm presenting now the case
Where someone's giving you a black and white image.
Okay, so, not axes, that's all.
Don't do… don't… any access is here, I'm just going to throw an image.
And, in this image, there is, going to be… the cub.
Okay, so this carbon's gonna be there. And, I want to tell you now the following. I want to ask you the following question.
Can we do something like that in a two-dimensional space?
Obviously, if you… Know exactly the shape of this, thing, of the cow.
You will do exactly that. You will go around, position this template, right, around this kind of image, maybe some kind of systematic fashion, maybe we'll start from the top left corner and go around, and
Definitely at the location of when we have some overlap, or strong overlap, that couple… that correlation.
Because you're correlating the location with this… the other kind of, the signal that you get, the image in this kind of case, you'll get a peak, and that's basically what you report.
Having said that, Is there any other thing that you can do if you don't know?
The specific shape of that vehicle.
And the answer is yes.
What we can do is, we will,
instead of actually having a template of the vehicle since we do not know it, right, we will adopt
a different approach, we will shrink the template to some primitive shape. Okay, so I'm suggesting now a primitive shape that I think will be useful for this type of classes or vehicles here.
First of all, to understand the contents of this kind of small template, which is 3x3, I should explain also a couple of things. Over here, we have
an image, right? We have an image. Now, the image obviously consists of pixels, as we mentioned before. This is a black and white image, so every pixel has a dynamic range.
What is a dynamic range? As we mentioned nothing earlier, typically we have 8 bits of dynamic range, okay? In other words, I'm representing the color of a pixel with
A binary number, but it's from 0… or an equivalent integral number, from 0 to 255.
255 is a complete dark, a black.
And, two… sorry.
Zero is the dark, is the black, and 255 is the white.
Okay, so over here, we have numbers. Basically, in a single-channel kind of image, it's as if we have a matrix in front of us, right? Some elements of the matrix are 0, some other elements are 255, okay? And
Or equivalently, someone may go and divide everything back to 55, and have numbers 0 or 1. So, one and the same thing. In fact, we typically divide by 255.
And, so if we have this, situation, can someone suggest some
Population of this template over here, that it will detect something in this image.
some population.
Where am I, please? Occulation of numbers.
So, I'm in a look at this thing.
The middle one should be bigger. Yes, so this one, if this one, is, shot off my,
So the white.
Right?
I'm gonna start again from this, location over there, and I'm stuck correlating. Correlating, I'm gonna move over here, and so on and so on. At some point, this thing
will be right here. Do you see that? It will be right… right there. At this moment in time, you will get a fairly strong religion.
And the same thing will happen on the bottom of this vehicle, on the floor of the vehicle, and in the hood, or whatever you want to call it, okay? But someone may say, you know what, I think I need
More than one.
of these kind of small templates, called patches, that, I can program, for example, like this. I can program, like.
I can have the ones over there and the zeros elsewhere, and this probably will pick up some features over there.
And, and, and so on and so on, so I can have another one called, like this.
And, we'll pick up some features over there. And, the wheels, I can,
Sort of, program something like this.
So, we kind of reached the following kind of conclusion that we need many patches to pick up features out of an input image, in general.
Especially if we do not… I mean, we know something about the class, obviously, over here, and we can intelligently kind of select Apache. But the direction we are going, as you're probably guessing, is that someone
Will… With some procedure that we are familiar with already, populate these numbers inside these patches for us.
Okay, so this, this, remember the automatic feature generation that we saw in the dense network, kind of, earlier? This,
features.
are going to be picked up from correlation of appropriate numbers in this template, and this network, we'll see why we call it a network, this kind of thing is called a correlation neural network, or a convolutional neural network. In fact, there is a system operation called convolution.
That, it can be,
I think I went to another course for this image.
Where is it? This guy over here. But if you… If you,
sort of, if you go to an implementation, like, in a kind of framework and so on.
The two operations are more or less equivalent, but you get some kind of flipping kind of effect. We don't have to go through all these kind of details. But if you go through an implementation, most of the implementation, if not all of the implementation, are doing correlation. They're doing, effectively, a third product. Let's see how this is happening over here, the example with a taxi.
This is the image that I gave you just now, the black and white image, right? With some kind of object in there, right? And then here is your parts. In my example, it was 3x3, here we see a 2x2.
Gosh.
the batch has some numbers, the image has some numbers, I'm taking my batch, I'm…
Locating over here to start with.
And I'm doing what this operation is called. A dot to the product. Am I familiar with the not product? Have I done this before? Obviously, I have.
I move it in one location to the right, I'm doing another dot product, another dot product, and so on and so on, so I'm getting a new
image. I'm using a new image. We would be calling these things featured maps.
Maps, because they are definitely mapping features around, we have this kind of operation, we know the location of features that we're picking up with this correlation operation.
And, we are definitely going to call these guys. Sometimes we'll call them images. This is at the very beginning of this,
network, which are raw data, or we'll call them input feature. So, in general, we'll have an input feature hub processed by correlation operation.
Produce an output feature.
One thing that is actually very evident here is that,
Is the output feature map going to be smaller or larger than the input feature map? I don't know why it's going to be smaller, because… because obviously, the only way that this feature output feature map will be
equal to the size of the infographic map if this path was one by one. Obviously, we have these examples. We have…
We have, elements of these neural networks that, we're going to study that have a patch of size of 1 by 1, and this may not make exactly the same, an intuition right now. Yes, go ahead. Can you take feature maps of inputs that have time, access?
Yeah, so, for example, we have these type of networks that are called one-dimensional convolutional neural networks that are able to do the same operation across the time union, exactly the same operation.
Is that what you're asking? I think not. It's more like, if the actual input changes over time.
Well, okay, when the input is changing over time, in a sense, we have, let's say, some kind of a sequence of images, or some kind of video, we will know, we will learn how to process them. Typically, what you process them is you're processing with a sampling rate over time, but you process one image at a time. And, you stitch together predictions.
We are an odd don't think.
That is, it's called a probabilistic graphical model. So we'll learn after we go about two simple computer vision tasks.
We will go and do the so-called tracking.
I've got Darcy.
Okay, I don't think I have anything else here. And by the way, this kind of operation we have done over here is not something formal. Even when you go into a course, I think there's a course here in NYU called Computer Vision.
Right?
They teach this type of classical computer vision tasks, I think, like filtering, things like that. These were known for decades and decades, and everyone who has worked with photography, they must have come across this type of blaring filters, and these filters, and that filter, changing the… sort of processing the light a bit differently than others, right? So it's exactly the same principle.
That we have over there.
Okay, so I don't think I have anything here. So let's look at this, let's look at now the…
So, remember what we have just discussed. We need multiple of these patches that we will be calling now from our own kernels.
to do this operation. So if I stack these kernels, I create what's called a feed.
How many filters… how many cameras I need in each filter, we will see. I mean, in general, I will need…
plenty of kernels, but it's a kind of a design factor. It's not something that is uploaded and we prescribed here.
Evidently, I, I, I will come to that, here's the picture.
later. This, shows, some form of,
Some do other kind of parameters in this kind of filtering operation. Here, the filter position was just one touch.
And the white cells over there are… the white pixels over there is the original kind of image. And you notice,
Two things being ordered over here, the so-called stride.
Where, for example, the stride of… that we walk with, so if you make a… have a larger kind of stride, you are covering, faster, if you like this space. Here, we are skipping the stride of… we are skipping this pixel.
So the red thing finds itself in the dotted blue thing after just one
One, one location to the, to the, to the right here.
And obviously, when the striders won, we go and operate with this specific 3x3 kind of area. We don't skip, okay?
The other, obviously, the skipping has some effects, okay? So the stride has some effect, so we'll see those effects in a moment. It will also save some operations during the correlation operations, as you can imagine. The other thing that's actually going on here is…
This gray area on the image. And this,
This, area is called padding.
These gray things are typically zero values, and the reason why, actually, we are doing it, there are two reasons. Remember the earlier discussion that our output feature map is going to be smaller than the input feature map?
So, when we have funding, you will see this kind of animation.
later on, when we have padding, as you can imagine, that filter operation results into a larger input-output feature map than when we don't, okay? And the second kind of byproduct is that,
As the capital is located over here at the edges.
The kernel has always kind of a fixed way to pick up correlation to what is actually located over here at the edge, but if we have padding, the kernel will move in various locations at the edges of the images and pick up additional features. So it has more flexibility in picking up features at the edges, right?
You can see, like, in an animation, a portrait.
For example, over here, while before, the kennel would actually only go over there, and that's it at the edge. Now the kennel can correlate with its
more rockets of its own kind of rows and corns, the last picture, the edge picture of this kind of thing.
Okay. All right, let's move on a little bit. Now, what I'm going to do now, next, is to answer the following question.
What is this… operation,
you know, in a bit more, general kind of sense, because our images are naturally coral images, so we have how many channels? We have 3 channels, red, green, and blue, right, let's say. And of course, we have other applications where the channels are much smaller than that, like satellite imagery and things like that, and we have potentially 7 or even
More… higher number of channels.
And, therefore, we have to understand this operation in the… when we, in general case, we receive the volume, right?
And we deliver another volume, right? So we receive a volume, do the correlation, to pick up spatial features, and we deliver to the layer above us another volume, right? And so I'm going to draw something that does that.
And at the same time, out of this discussion, we'll understand also what is a convolutional neuron. Okay, we saw the sigmoidal neuron, we saw the other neuron with the LU, right, and the dense kind of layer there. Now we will see what is the convolutional neuron, okay?
let's, let's start a drawing. So I need some kind of realistics, to draw this kind of diagram.
So I'm gonna start at the bottom over here.
And I have my input feature marked right there. So this is my input volume.
Obviously, the input feature map has A number of pixels.
Yeah, I'm just drawing some pixels over here.
And, has a depth.
off.
ML-1.
Has a height, L minus 1. It has a width.
L minus 1.
And L, smaller than L there, is just basically an index for the layer that I'm in. That's a general kind of thing. So this diagram applies whether we are at the very input or at any level in this kind of network.
So we'll do some kind of operation over here, this kind of correlation operation, and definitely we are going to define
An output feature map.
And as we discussed.
Using a 3x3 filter, the number of pixels of this kind of output position will be smaller in general than the input fission.
pixels. So, I am going to suggest that if I have a specific, and the headline of this diagram is the snapshot.
of CNN.
Layered.
Corporation.
we will observe at this specific snapshot in time what is happening. So if I have, let's say, the filter over here.
I have the 3x3 filter, in other words. The question for you is.
How many kernels do I pack into this filter for this specific input feature map?
In other words, what is the depth of the finite?
The filter will also be a volume. So the question is, what makes sense for this volume to be?
Okay, at least I'm water.
Well, there are 3 options. The filter may stick out.
you know, deeper, will be deeper than language.
The second option is that the filter will be shallower than the input feature map, and the third option is that the filter will be exactly the same depth as the input feature map.
Which of the three makes more sense?
So if it is deeper than the input feature map, right? You know, there's no reason why it should be deeper, because
You know, what are we going to do with the additional chemos of the filter? There's nothing to correlate.
So, when it is shallower, that also does not make any sense, because,
although it can be shallow work, someone could ironically call it a shallow kind of filter there, we're going to leave money on the table. I mean, we're going to leave some channels, right, of the input feature map where don't pick correlations there, right? So the only reasonable kind of answer is that the filter is going to be exactly the same depth
As the input feature map.
That is the only reasonable choice.
Okay, so we kind of designed the filter now. We know the number of,
kind of gears that we pack. And obviously, as you saw in the previous kind of figure.
at this specific snapshot in time, if I do the correlation.
I will create a vector or a scalar.
What am I correlating now with, first of all?
If I call this guy this… if I call this guy, X.
So I have,
have an extensor, which is a volume. Because are we also, like… so when we take a correlation, we usually, in the 2D case, we saw that we are summing up across a two… Summing up? Yes. Right, so here, are we also doing it across the table? Yes. So we get a scale.
Then we'll get a scalar. A scalar. We will always get a scalar out of this dot product, right? I mean, now the dot product is in a volume, a dot product, not of a two-dimensional dot product we saw, it's a three-dimensional kind of dot product.
We'll write the equation now shortly. And, let's assume that the filter is coloring something on this,
location, which is I comma J.
Has some spatial coordinates, okay?
There are two things going on. Special coordinate.
I comma J across the height and width, for example, so this is basically the HL, and this is my WL, right? And this is…
And that's ML.
So, if I take this kind of column, and rotate it 90 degrees, And,
give you the pixels in this column. How many pixels can I have in general? I'm going to have
ML minus… sorry, not ML minus one, ML pieces.
Because the output has that ML, right? So we've got ML pixels in this kind of column.
And, I would also define an index called KL,
Which is going to be definitely between 1 and a mL.
To be able to answer the… Question.
To be able to answer the question as to what?
pixel is coloring right now this filter, right? Which one?
So I can do the same, by the way, for the filter over here. I can take the first kernel.
The second kernel.
the last kernel is here, so I took the filter, and I rotated 90 degrees, and I just write explicitly
draw exclusively with each kernel, so how many of these guys do we have? We have ML-1.
All of these kernels, as you just told me, are coloring One pixel.
One pizza. You just, you just told me. Okay, so my, my question to you.
Is that, obviously, this pixel will be located somewhere.
Let's assume it is located over there.
So… If I do write the equation, the three-dimensional kind of correlation, which I promised, So, given.
I'm writing given, I comma J, Comma. K. L.
during this kind of period. Before, I had… summation over.
the indices V and U.
that are picking up numbers from the… from the kernels, right? So, all of these kernels are indexed by U and V.
So… what are we correlating with? We are correlating with the overlapping X.
Only the overlapping volume.
with a filter will be, here. So this is basically I plus U, J plus V, KL-1.
Because the KL Manage one, I forgot to mention, is the second index.
that I can define here to tell me about the depth
Of the input feature map, times. I'm gonna call this guy…
I don't want to confuse with… with the width over there, so I hope you… I'm overusing some notation here. I'm going to call it W, because that's typically the parameter kind of vector I used before.
And this is U comma V comma Kl, KL Management.
So this is… This is times,
W, so this is X times W of U, V comma K, L comma K, L minus 1.
And I claim that this is going to… a lot…
a pixel, but I need another summation. I need a further summation.
Because I need to sum over
All of the kernels, all of the kernels are creating this green pixel, not one, all of the kernels, right? So I need to sum over what?
Data minus 1.
I'm summing over K minus 1. I'm taking its kernel, Correlating with overlapping.
area of the X, right, of the corresponding X, and I'm summing this to the next dot product to the… so I'm doing this three-dimensional kind of dot product as
KL-1 summations of a two-dimensional dot product. This is basically what I'm doing.
Which is exactly what I did earlier, but only had one depth of 1 back then in the previous kind of figure, right? In the previous figure, I wrote depth 1, I got Kl minus 1 of these patches, which I have to produce a dot product, right? And I will sum these individual dot products into one number that is going to be called
That… Z.
So I'm writing, so given…
I'm forming, give a comma, I'm forming a Z,
Conforming a Z of I comma J, comma, KL. That is going to be… this is going to be my…
Should've… Brilliant picture.
which is going to be located at the i comma J spatial coordinate, Somewhere, inside this clothing.
Okay, so wouldn't the depth always be 1? Because the depth of printer is ml minus 1.
Okay, this is what I, I want to, I want to,
I want to do it for status.
BFI…
take this and see what happens in the next snapshot. Let's assume that I move this filter to the left.
This is a question I've ordered to answer.
If I move the filter to the left, do I… Lot.
You will determine the red pixel, or something else.
If I move the filter to the left, What, what's gonna happen?
Remember, I'm changing the special. Yeah.
coordinates, therefore, I am plotting which pixel.
And all of the ones I have shaded with red.
But I'm blotting.
Another pixel on the left of the green pixel, in the same Death shooting.
So, back to your point, One filter can produce one slice.
of the output volume. One slice, and only one slice. So let me delete this thing that I plotted over here.
Got it.
Let me delete this color, too.
Which means that in order for us to produce an output volume of that ML, I need more.
More filters. That's the only reasonable answer. That is really the safe conclusion of the discussion. I need to define multiple filters, right? So,
So, I need…
ML filters.
performed.
a volume… Awesome guests?
ML.
Is this operation legal?
Only one game.
What we have discussed about the… need for…
When we look at the dense networks, we discuss about the need for introducing nonlinear projections, right? So, over here, we're trying to pick up spatial features that are present in my image, right? I need a nonlinearity.
I will then need to follow
from the dot product, but usually what comes after the DOS product. A non-binary shall maintain the relu.
After the nonlinearity, I'm going to add a bias and pass it to the relative. This construct of the linear correlation operation in the three-dimensional space of the volume filter, this filter, which obviously has depth and minus 1, with an input feature map.
This operation, is what is really happening inside the convolutional neural network, and
What can I call a convolutional neuron?
This operation that we just saw, that produces my Z, so this is my convolutional neuron, is going to be this kind of operation, that I'm going to take the input feature map, which I will… I will call this X,
I'm gonna do the 3D.
correlation.
I'm producing my Z, my Zs, which is basically a slice.
If I use this filter…
And, after some kind of bias addition, which I forgot to… Wash off.
Add here.
I can pass it through NL loop, to form an H.
Obviously, the age would be like a matrix. It would be a matrix. It would be one slice here for you.
Yeah, sure. So, just to make sure, this depth…
Because what the other one was saying is just, like, why the…
And the depths are different between what we get and where we start, because the, ML…
depth to, like, the slightest to me, I thought, were, like, the alpha channels of that photo. But, like, ML is the input and output M, capital M, designates the input and output depths, which is RGB, for example. Sorry? So it would be, like, RGB, for example.
No, for example, if you're looking at the very first layer of the CNN, the input depth is prescribed by… it's going to be free.
But obviously, the output depth can be more than 3. Okay? And of course, as we'll see, the usual pattern is the deeper you go into that network.
the depth will increase. That's a typical part that you will see in the CNN. So, by the way, the ML that we have over here is for us to define.
ML-1, ML for us to be fine. Except from the very input where the ML-1 will be described by the… by the image that we're getting, how many times it will have.
So this is basically our, convolutional neuron.
And similar to what we have seen earlier, we need multiple neurons to form an output feature model.
Okay, so we need to effectively have, something like this.
So here we have our…
Our finger, that is correlating with the input volume, which is the blue thing over there.
Notice the depth of the figure in the sky is going to be equal to the depth of the input volume.
And… The output will be one slice of the green void.
And I need, for D out, F, D out, I need multiple figures.
Multiple of the strengths, hang out.
Guys, to do that thing. If you understand this, that's basically everything that you need to understand on the simple convolutional kind of neuron. Obviously, we have, one of the parameters in any API call will be the number of neurons, like, you're gonna have…
The shape of the spatial dimensions of the filter, let's say 3, comma 3, followed by, or preceded by, a number in the number, which is the number of neurons, which is, of course, will define the output depth of the output volume.
Let's see an animation here to see this a little bit more concretely what's happening.
So I have here… And input volume, which is padded.
And, it has definitely death-free.
And I need to form an output volume which has depth 2.
How many filters do I need?
Two filters, because the output volume depth
You know, to… one slice of the volume will be the responsibility of one filter, okay?
So, for a volume of F2,
I need two filters. This is the first filter, this is the second filter. Notice in the animation that filter 1 participates only in the first slice of the output volume.
While filter 2 participates only in the second slice of the input.
Of the aquaphone.
So this example, I think it's a good example to just give you,
sort of that, emphasize what we just discussed. What we attempted to draw is one snapshot out of this guy.
Alright. Is there such a thing as, like, like a hidden… Like, here you have…
two filters. We have three filters. You only have two, you only need two. Is there such a thing as, like, combining two filters in here, which is just pretty much, the two output volumes, essentially having, like, a hidden kind of, like, layer with these two filters.
Or some are one filters and some are two filters. Do you understand what I mean? No, I don't.
Okay, maybe, maybe you can follow up, but, you know, definitely what we see here is, what we do. Obviously, we have a bias, right, that we all should do.
the numbers that we see here, so we may want to verify this as a just simple numerical expression to get the corresponding number over there to see what's happening.
Any questions?
Yes. When we are, creating an output volume, right? Yes. So, the depth of the output volume does not depend on the input volume, right? This is something that we are deciding to do. No, we decide. Absolutely. This is something that is on the basis of our design of, the filters, right? And the number of filters we… I'm just trying to understand,
the number of filters that we choose is essentially the depth of the output volume. Absolutely, absolutely. Two… two figures…
two slides are there. And including a lot more filters essentially enriches the output volume, and we need to understand, you know, what is the intuition behind, you know.
the pattern I just explained, as you… the deeper you go into the CNN, the more filters typically you see, so we'll see an example in a moment.
So, let's see,
Let's go now to this obstituent battle discussion. Actually, I need to show you this plan. That's actually the next discussion.
But before I go into this kind of discussion, I want to highlight some formula.
Here, but you need to write down.
And this formula is giving you the input and output feature map dimension… sorry, the output feature map dimension, in terms of the spatial.
dimension. It gives you, in other words, the HL and WL, okay? So the HL is HL-1, plus 2 times the padding.
Minus the kernel size, which would be used 3 for the kernel size, divided by the stride.
You take the floor, how is it?
In your pest one.
This one will give you all the information that you need in order to draw the output volume, because you know the number of filters, you know the spatial dimensions, always exactly the same format will apply for the width dimension.
Right? And, so… -Oh.
That's basically what's… what's happening, okay? Now, you… you know that…
I'm gonna… I'm gonna, pass, some of this kind of discussion And come to this,
To another component that we see very regularly in convolutional neural networks. This component is called a pooling layer.
It's actually very difficult to see
This kind of data being used as kind of a distillation mechanism. Yeah, I'll explain what it does.
So the pull-in layer, instead of creating this kind of dot product.
Against lines this kind of kernel around.
And he's applying a function.
This function would be averaging function, would be a max function. Typically, we fill out the max pooling by max function. Effectively, if the contents of the email is 1532 at a specific location.
where the kernel is located, right? It will…
It will pick up the maximum
Number out of this, and provide data to the architect.
And what is the intuition behind it? If, after, let's say, correlation, I have, let's say, an output feature map, I want to perhaps propagate the most
important or strongest feature out of this output feature map. Leave everything else behind.
And then this will be my… a kind of a distilling from perhaps some kind of a noisy situation I have after the correlation to something that was a bit less noisy. So, we are trying, effectively, to improve the signal-to-noise ratio in a kind of a max pooling. However, at the expense of
information dose.
And this is the reason why Max von Layer has been attacked as a… not the best thing you can do, but still is quite dominant because of its simplicity in many architectures.
So, we'll see the, typical kind of network architecture now, which consists, again, from
these kind of correlations, that we just discussed in each… in the single layer. But before I go, I want to just also cover another component, which is very difficult to see also in some architectures, the one-by-one convolution. And many people say, okay, what on earth are we trying to do with a one-by-one?
And if you see the three-dimensional…
If you see a three-dimensional clock of what's really happening.
First of all, the 1x1 is the only way that you can create an output slice of exactly the same spatial content as the input. And as you can see here, at every location.
We are progressing, we are… Combining all the depth information we have from
So, for example, someone may say, hey, this one by one seems to be a robot also to be used towards the end of the network, as a kind of termination of the network, or we may also use it in some instances, to do some kind of a matching, dimensionality kind of matching. We will use it inside, for example, rest nets and things like that as one of the components.
We need to do so. Do some kind of, dimensional technology.
In the question with.
Okay, so that's basically the three components I want to mention for now.
And, what we will do next is we'll go back and see some visualizations of this kind of an architecture.
Okay, so… This is, like, a toy kind of, figure. I told you that,
nonlinear kind of, spatial correlation will happen over here, followed by, sometimes, with some information distillation, because of the polygent layer, and so on and so on. And, at some point.
As we go up, and better to show this image right now, as we go up.
We will reach a point where, since every time that we use a 3x3, let's say, filter, we are slinking the spatial dimensions, and we are staking, slinking, stinking, at some point.
We'll reach a level when there's no point to continue to do correlations.
to the future. We are run out of dimensions, no matter what we need.
2 years?
Especially a dimension of the output feature map of center by cell.
So what we do to terminate the network? We say, okay, What?
Obviously, we have a volume, here, so we have some specific depth. I'll explain what happens to the depth.
We will take this, and guess what we will do? What's something we've seen a bit earlier in the previous lecture. What we did is we will flatten this.
Right? Create a very long vector, out of it, and gradually feed this vector over a combination of
Projections? That are dance projects.
To forms.
The posterior probability distribution of a thousand The machine by one.
And, every time you see a function by one.
a specific dataset classification task from a specific dataset comes to mind, like ImageNet. ImageNet is a dataset which is… has a thousand classes. It's not necessarily to mention it here, but yeah, whatever number you see here is going to be obviously 10. For example, we saw an example with 10.
In the previous context, it would be 10. That's how we terminate the network. We have some form of combiners, nonlinear, nonlinear projections.
Okay, so the question is, okay, we have an input image. I told you that in computer vision, the images are not really very high resolution.
And definitely we… the computers are able to distinguish features that we cannot, given. So it's, the input resolution is very coarse here, 224 by 224, 640x640, it's these type of numbers that you would see.
And, the image, obviously, is coming in with, let's say, something a quantity of 3, depth of 3, right? And, in the first layer, we are using 64 filters, or whatever the number is. Then we increase it to 128, 256, 512, and we stick with everything.
And the question is why the filters are… Sort of increasing.
And the answer to that is, the intuition is the following.
If you go and see what happens inside the CRM.
Something that is going to come up as Assignment 2.
In your coursework here. I,
If you open up the guts of the CNN,
Let's say after training, right? What you will see is that the very first layers are
sticking into, or converting into, detecting simple features that are, especially if you see the feature maps that they are creating, they still have this kind of geometrical information around, right? Like edges.
You know, this type of stuff, like, and so, the deeper you go.
the features that are being created are becoming more and more abstract. They are losing their…
geometrical, nature, right? And they are becoming kind of,
artist… a lot of… if you see the realization, you probably see some kind of art of some sort in this, in this kind of images, right? They are… so…
the deeper, however, you go, the creatures… you have the greater need to combine, in multiple ways, the information that it is provided below you, right? So, you need more filters, more combiners, nonlinear combiners, effectively.
to produce an output volume, and the output volumes. And don't forget also that the…
special degrees are shrinking at the same time, right? So the, you know, over here, you need 512, right, of this kind of filters, or maybe over here. You need 5-12 of these kind of filters to combine information which is actually coming from
from… from this kind of infrastructure, right? So you're… the deeper you go, let's say intuition, you are trying to create as many correlations as possible, as…
possibly, up to, let's say, 5 to 12, kind of euros over there to…
Do they have anything that these previous layers can actually offer?
That's basically the pattern that we see very frequently, and of course, a button exists.
Only to limit.
not broken, but we see other things as well. Sometimes you see, networks that are picking information even from other
earlier kind of layers also. It's not only necessary that information will be provided only from the… from the layer just below it. Sometimes you see architectures that are crossing.
The layers for some subsequent kind of crossing is being picked up from a variety of
area kind of players, not only just human. That's some architecture that I hope to see a bit later, when we are looking at the auto-emporal type of architectures.
So let's look at an example of a CNN.
In some kind of operation.
So, over here, you have, a very basic…
Binary classification here, as simple as possible computer vision that you can do.
And the two glasses are patent doors.
Obviously, the images are coming in a variety of shapes, and you have to do certain pre-processing. In fact, preprocessing is…
Over here will be limited to resizing.
So you size the images all to be within the same kind of size, and this is basically the…
and this size is almost kind of hard-coded in this kind of architecture, right? So, this is one of the things that we'll release a bit in the future, how we can actually have architectures that can accept multiple sizes.
So over here, you see the…
the classical pattern, convolution, max cooling, max cooling, and so on. Notice the increase in the number of filters as we go deeper, up to a point. And also notice, the,
what's happening in terms of architecture. You have a relu, you have another value in the second kind of layer, third layer value, and at this moment in time, you have, we'll see a bit later, some kind of a special extent in my output feature map.
Some, and then you do the slot again.
And some… to form, to use food.
projections over here to death layer. Obviously, the last line shouldn't be a surprise to anyone. You need one neuron to combine all the features that are given to you by the previous death layer, and obviously, you have to use single model, because it's a binary classification test exercise.
So that's basically in terms of the architecture, which I think is pretty straightforward.
One thing, however, is,
can be highlighted a little bit by looking at this kind of architecture that I want to mention here. Look what happens to complexity.
What are the number of trainable parameters that we have?
inside the first layer. Can someone tell us the parameters in the first layer? We have 3x3, right? We have 3x3 with a special degree. The parameters are all in the filters, right? So we have 3x3, the spatial degree of one filter, and we have,
So we have that assignment.
We have a 3x3.
by 3?
3 is the input, dimensions, right? The, the depth, the input depth is coming with,
3, right? And we have, 32 of those. So we have 9 times 32. How much is 9 times 32?
9 times out of the two is, Must be.
from my nose.
9… times 32.
is 288.
And, we have, So…
No, something is… something's wrong here.
So…
For Manny would help.
Okay, let me do that.
So we have an input volume, blood counts.
With a depth of 3, right?
This is DEPCO3.
Right? And we have, A 3x3.
Filter, right, with depth of 3, so we have 9 parameters.
So, no, we have 9 times 3, so 27, so we need to multiply this by 3. So this one will be 27 by 32 filters, so we have, you know, 27 parameters.
a filter.
Time set the two filters that we see from the…
on the API call over there. This will give us 864 parameters inside that filter, but it's only for the filter. However, remember that we follow the dot products with bias, right? How many… what is the length of the bias vector?
32, because each correlation will form a scalar. So we have 32 scalars, so plus 32 for bias.
The sum is, 896 parameters.
And cut out 96 biomes.
So, that is basically the cost of the first layer.
Evidently, the cost is increasing because our depth is increasing.
Right.
However, look what's happened.
The special feature that we are forming as a vector is 6,272 elements.
The first dance player It's costing us 3.2 million.
So, out of the total 3.4 million parameters, 3.2 million parameters is the test they are responsible for.
Everything else is, like, much smaller than that. So, first of all, If we can,
Kind of make an omen over there.
Convolutions are… not necessarily very expensive, okay, at least in this,
part of the course are not going to be necessarily very expensive. What is really expensive is the dense layers in this kind of architectures. And one thing that you can note here is that if you go and train this thing, I mean, how do we train this thing? Exactly in the same way as before.
We are forming some, binary, we are feeding the white hat with binary percent, probably, right, loss. And we are using a stochastic-grade descent, or a Casino stochastic-grade descent, to actually minimize the loss.
And okay, we're doing this training over here, and
And we observe, like, this loss as the number of epochs is kind of increasing.
Do you have any comments for this behavior right here?
Have you seen this earlier?
As we progress the training, the difference between
Validation and training loss is increasing. So we have a feeling. Was there any surprise of this behavior?
Well, in order to answer this question, we definitely need to look at the data that someone gave us.
And the data someone gives you is, if I remember correctly, is going to give you 2,000
Cats and 2,000 dogs. And we are hitting this dataset with 3.4 million parameters.
Definitely there is,
Overfeeding, and so if someone is giving you a much larger kind of data set, do you expect the same behavior or not?
No, no.
So…
In computer vision, we have one tool that we use over and over again to increase the number of examples artificially.
that are being considered during the training. And this is called data augmentation.
So, data augmentation is very, very important in computer vision in general, because of this type of effects. So, for example, what kind of data augmentation we actually can do? If you go to the PyTorch API or whatever other API,
You have a very special section of…
Transformations, and of course, there are also several kind of libraries
One is starting with Facebook, excuse me, aluminations? Let me find it, okay? I'll tell you the library, which is sometimes I…
I'm so scared.
data augmentation.
augmentations, okay? So this is basically the one library.
That's one library. And the other library is called Cornell.
Many of these kind of data augmentations, are taking, substantial, processing, okay? Because many libraries do not… do not use the GPU for data augmentation. Cornea does.
Okay, so there is, you need to be careful what you're doing, and, you know, sometimes you… one library performs better than the other, that's the issue.
Obviously, with the documentation, we do these transformations, we are sort of sometimes creating, dropping some channels, for example, creating grayscale images for the same object. We are,
Going to be creating some really ugly images of cats and dogs.
Having said that, the network is benefiting from such augmentation, because we are improving the number of labels
That we are hitting the network with 3.2 million parameters, and…
Although this is not also a very nice behavior, certainly it is better than we had earlier.
Okay, in terms of, overseeing.
You have some strong variability in terms of losses, other things that we can do in order to improve that variability.
Okay, so one thing I want to, close this kind of discussion is,
You know, this kind of notebook is,
going to treat, if I remember correctly, a similar kind of problem.
But this time, This same network will reveal to us what this is doing.
So… I want to stay on a couple of things that I mentioned with Terry. Look what happens.
In, outputs of these first kind of layers, right?
Versus later kind of layers.
As the features are becoming more and more abstract, from a little bit more geometrically oriented.
Almost kind of popping the shapes of the objects that they're seeing to more abstract features that are going to be combined in it.
larger number of ways, as I told you, with this kind of number of filters, to produce, finally, the vector.
Age that represents, for a given input image, the concept of the cut of the dog to do the job properly of classification.
Definitely, we can also observe, what the filters are actually doing.
We are able to…
suggest that the… remember the patterns that I was drawing in the first kind of taxi example, where I was telling you this would be black and all this would be white? Evidently, we don't have this, sort of… such a clear kind of patterns there, but you can actually definitely observe some lines going horizontally, some other lines going vertically, things of that nature, okay?
And we are… we're able to…
Sort of. Another thing that you will investigate is,
I can't agree 100% sure if we're going to investigate that.
but I'll tell you in the assignment, is that you are able to provide some kind of,
With certain techniques, some kind of a map where you highlight where The network paying attention to.
In the… projected in the input image to do the facilitation.
Okay, so over here, the classified elephant.
That sort of, network pay attention to this area.
But that's sometimes useful for debugging purposes, as well, as you're also persuading yourself that, you know, this thing is doing something, that is kind of intuitive as well.
And now… I'll close this discussion on this kind of architectures with
We're presenting a network, which is called a similar kind of network, that, solved, quite a lot of problems in, and, and is used today in, as a bread and butter for virtualization. So you will see it everywhere, more or less.
This, comes with the name as ResNet, and I'll explain why your design… what the design of skip connection is causing to the output prediction, and so I can only understand its, kind of impact there and value.
So, chances are, in this course, you will use it at some point.
in an assignment, or we will see that over and over again in object detection, stuff like that.
Okay, so let's look at the…
discussion as we start to residual.
Mr. Happy.
more… notebooks.
So, I want to start this kind of discussion by presenting, going back many years, and
Nope, during what was happening around 2014 timeframe.
But basically, at that time, you know, it was well underway.
The premise that the deeper that you go, the more linear you add.
the better the performance. And at some point around that, timeframe, they observed some kind of, floor.
or ceiling, rather, on how the performance can improve. So they were not able to scale neural networks more than
let's say… I can't remember now exactly, but it was, something on the order of, 16 layers. Okay, so we'll see an example of, a network called,
BCG from the original kind of names of the authors, that it was maybe 16 to 19 layers. And the reason why they were not able to extend it is, as it turns out, gradient flow.
Okay, so gradient flow was problematic inside this network, and the rest nets solved this kind of problem.
So the reason why there is a significant kind of jump
to performance, with the adoption of kind of presence is what we're gonna do now. Evidently, networks, to have nowadays Snext, and all this kind of other stuff.
The essence, as a concept kind of evolved.
But, let's see, at least now, the…
baseline kind of lesson architecture and understand why gradient flow improved significantly, and what is the reason why the gradient flow is such an important, I guess, component in gradient.
Alright, so let's, plot here.
a network. It's gonna be just, three…
What we call it as, blocks.
So, I'm gonna have… Knock F1.
Followed by… knock, F2.
Nope.
They could grow it a little bit smaller.
We're gonna do exactly similar to whether it's smaller.
So this is F1.
And, what the… Microsoft Researcher, did.
suggest is that if you take the input and you add it to the output over what is known as a skip connection.
Good things will happen.
We don't understand what I'm looking for.
So, if I go now to another state, F2, and do exactly the same.
And finally… I'll go and plot F3.
And do exactly the same.
Let's informed.
I'm gonna call this Y3. I'm not gonna put the hat.
I could… I could do the cutest one, but…
ambition is slightly different here. Y0?
Why once?
Why two, and finally worth it?
So the equation, I mean, this network is implementing repetitively an equation called YI.
is equal to FI.
YI minus 1.
That's why I mentioned. That's the equation.
I hope you're living nicely, right?
This is the equation.
But, for each block that I see, that's the equation where the street connection is. So I have a task while isk where the strict connection is coming.
Okay, so let me write down now the… output.
My aim is to write down the output only as a function of the input X0,
And, the name… and the blocks F1, F2, and F3.
So I'm going to write the table as Y3, just like what the equation is suggesting. It is F3 of Y0… of Y2 plus Y2. I hope you agree that's the first thing. I can then replace Y2 with
F3.
of F2.
of Y1 plus Y1.
plus F2, Y1, That's what I want. Do you agree with this, or no?
I just replaced… Y2 is, the cheap one. And then, finally, I am going to arrive.
Art.
F3.
of F2.
I have one of YZO.
dashboards here.
plus F1, plus Y0 plus Y0.
plus F2.
of F1, of Y0.
That's crazy.
Thus. Thus.
F1.
Why zero?
That's why zero.
I haven't done anything intelligently here. I just wrote the equation.
by substituting two times the… What is, Y2 and what is Y1?
So the question basically says now that on the left-hand side, I have Y3, on the right-hand side, I have only Y0. And the functions F1, F2, and F3. What are these functions F1 and F2 and F3 is obviously for me to define.
In the present architecture, definitely will consist of convolutional layers, as we saw them, and max pooling layers.
And let's table that. It doesn't necessarily mean that F1 is just one layer, it could be more than one layer, but we just logged together what we call F1, F2S.
Alright, so I want to color now, the… this guy.
I will call it it… I will label… I will label it A.
This guy, I will label it… B?
And, screw this guy.
I will label it C.
And I want to plot.
This equation, to draw the block diagram of this equation, of this equation.
So I'm starting with A at the bottom over here.
So I'm having a YZO.
I'm passing it through.
F1.
adding… Why'd you got that one?
And, the output is going to pass through.
F2.
I'm adding to F2.
another F1, Y0 plus Y0.
And the whole thing, I'm passing it through.
F3, and I'm claiming, and I want you to verify that this is true, I'm claiming that this thing here
He's a…
Okay, so…
I started with the V step over here.
And this term is this thing.
And, I pass through F3, right?
The whole thing.
Cheer.
Right.
So, obviously, I have another two summations to do. Okay, so,
Definitely, I can go continually with the diagram. I see now the B. The B consists of Y0 that goes through
F1.
And then the whole thing goes to F2.
That is basically my therapy.
Do you agree that this is my beat?
And, I have,
F1.
Y0 plus Y0, and I'm plotting now the C here.
And, both… scene.
B.
an A.
I write it together.
informed by it.
Okay, so what?
Thank you. Thank you for the,
enlightening, drawing, exercise. Okay, so let's see what is happening in the so-called,
DTG architectures or convolutional neural networks of that era, before Let's keep on actions.
So imagine a network without those key connections now.
And, apply what we learned in, back propagation.
Okay, so I have a gradient that is arriving, let's say, at the output Y3, right?
It has to go through a string.
it has to go through F2, it has to go through F1 to reach Y0.
And as you can imagine.
The important thing is not to reach Y0, the important thing is to feed with a principal gradient all the…
W's, or the thetas that are present inside the F1, F2, and F3.
So what is your intuition? Is the gradient going to be…
Has an easy life to go through, let's say, 15, 16 of these guys.
As compared to the situation we have right now, which is the following.
This is all relative, right?
Let me color the gradient with orange, so it will go through.
And it will go through
You know, significant.
Much, much larger number of ways to reach YC.
So, in fact, It can, you can actually see here some paths.
that are… given a… directly connected.
The output and the input.
That is an example of what we call highway networks.
Where, we are sort of building this kind of, highways and roads where gradient can actually flow
Understood.
We see examples of other highway networks.
In, when we do some natural language processing.
Which one would you believe that is offering…
I will call it a better behavior of gradient flow. Just as a reminder, as I told you, if we don't have
enough gradient. What happens in gradient?
And if we don't have enough grades on F1,
It is as if we don't have F1 in our hypothesis.
And so on and so on, right? So that's obviously not good news, right? In the sense that we are expanding also the…
expense to have the F1 there, but we don't take advantage of it.
Of course, the gradient allows to explode. Remember, we made some kind of gains in power corrugation that were adjusting the gradient in a variety of ways. Exploding gradients are also problematic, because although we are able to keep them.
We have, no ability, to,
to control the training kind of process, even there. So, either exploring or division gradients are bad and useful things. So, what needs to happen is to have to be able to manage these gradients.
And we have a variety of techniques to manage them. One of them is to allow the gradients to definitely be going into
a variety of ways to all the elements of the network without exhibiting, for example, significant attenuation through just going through one path. This multi-path
A situation we have here has significantly changed. That is basically why the rest is solved the gradient flow problem, while before we had only one path.
forecast money.
What was happening in the… sections.
What was happening in the junctions?
Every time you see a junction, not here, but here, every time we see a junction, what is happening?
It is addition.
Remember the example with the first function that we saw?
There was an X plus equal, a DX plus equal expression, DY plus equal there in that single example.
So, I can write that, gradient flow
now has…
diverse.
Set.
of paths.
Right.
G'day, Deanne.
It's Loche.
through gates.
of worrying.
depth. As you can see.
you know, C is… so A is deeper than B, and B is deeper than A. I mean, this trivial kind of three-block kind of example.
Preparative flowing proofs.
There's a co- however, yet another thing.
which is, going to occupy us for the last 10 minutes of the course, because guess what? I didn't have a break, and I have to leave at 1.20.
So then…
The last thing I wanted to mention is something which is not so evident here, but there is one…
I will call it,
subdomain or techniques in machine learning and AI that it is very, very, I would call it, almost like a silver bullet to better performance, and this is called unsample learning.
So, on sample letters, let me write it down.
I'll explain what a sample learning is, the basic principles.
So the premise of assemble learning
is the following. This is also known as Committee Hardening?
Nowadays, there is a lot of, news.
About a mixture of experts.
Or MOEs, and people are getting really excited about this new discovery. Apparently, that's not a new discovery, but definitely the application space is kind of very new, so people are getting really excited. So, if I decide
to suggest the following, that my prediction, which I'll call WFAS Committee, is not a product of
Sort of… it's not abused by a single
Very powerful kind of predictor, but it is an aggregation
over, let's say, capital K predictors.
And, also, I know that,
These predictors are… shouldn't necessarily be very powerful. Let me call them weak predictors.
And, okay, sweet.
Of course.
Anyone has an issue? Anyone? Anyone? No?
Okay, let me go to the… I'll come back.
Okay, so…
So, what I was saying here is, I have this, kind of relationship here, some kind of, way to form
committees, using many weak predictors. To give you an example, I hope everyone here is familiar with somewhat with decision trees.
decision trees. So, decision trees are, sort of hydroponic devices, based on entropy, again, as a principle, to, do, let's say, either classification or regression.
And, when we manage these decision trees together, we are using a technique called… we are forming a technique called random forests. And some of you may have heard it, but, you know, the basic principle is the forming. It's very intuitive, the basic principle. If I have
if I want to form a committee of members, let's assume you want to make a decision, and the members of this committee is,
Coming, born in the same village, went to the same school.
went to the same university, had exactly the same courses done, and ultimately, after graduation, you call them to make a decision. You have 10 of these people in front of you.
What is your expectation about the committee? Will it be a good committee to form, or not-so-good committee?
not so good committee. In fact, you can say that in the limit.
even if you pick any one of these 10, and you call it a predictor now, to use in your decision, then it's one of the same things as having 10 or 1. And this is basically, if I may plot here a kind of a performance bar, really at the bottom over here is going to be a lower bound.
In terms of performance, which is going to be identical to, using a single… Week.
predictable.
However, as you can imagine, as you start introducing diversity inside this committee.
In other words, you find some kind of a randomization scheme. For example,
in Orlando Forest, the example I actually gave, you may, you know, one of the fundamental things that you do in a decision tree is to pick up at any stage a feature that you're going to split your data.
And there's a way to find the best feature to split the data, but sometimes you flip a coin and you don't follow the suggestion. You pick up another feature.
Distributed data, then you… Start…
Randomizing the behavior of a decision tree led to the other committee members.
Then you will see the performance significantly improving.
And, this is where…
But there is also an upper bound here, I just want to intuitively you to understand it.
Just like in humans, human committees, you… what you don't want, definitely you expect a committee to make their own decision, okay?
I mean, it's… it can't happen, obviously.
But what you don't want is to minimize that
Erroneous decision, just like minimizing the error quality classification kind of discussion.
And the way to do that is to avoid a situation where people are giving you correlated mistakes. In other words, they give you… they do the same… all of them are doing… or the majority of them is doing the mistake at the same time.
This is where the… the upper bound
In performance is… is… is met.
So that, that…
Some explanation about this, in terms of analytical kind of way of explaining this, is inside Ian Gutfeldas' book. I'm not going to expand on that.
But, definitely you can form diversity in a seamless possible way. You can have, take this data to predictor 1, another set of data to predictor 2, another set of data to predictor 3, that's the seamless possible way.
I can sample from a data set and give different subsets of the data to these different sequences of data, different examples as they met, to each predictor. So that's basically one way of improving on diversity.
On the data kind of side.
The second way is to…
randomize something in the behavior of its predictor, but what is meant in ResNet is to use different predictors. I can have a logistic regressor, I can have a neural network, I can have whatever.
Great. So, look what happens here.
Think of the predictors as predictors A, B, and C. You have three weak predictors.
Look what they have… look how they form their…
Hypothesis as a summation of their individual predictions.
Predictor A and predictor B and predictor C have different strengths in terms of prediction.
So, effectively, we can write as a headline that resonates exhibit some form of an sample learning.
Because of this diagram.
That's basically the concluding headline.
You heard me talk about, randomization methods. If you are interested, that's, linked up to you. No one's going to ask you. If you go to your, kind of,
either the two textbooks, you will see there some explanation of random forests, and that's a good example to use always as a… I will call it a randomization example on sample learning methods.
Okay, so that's basically it for, for, for today. I would, as I said, please,
Let the TAs know if you have issues with your assignment ASAP, definitely before Sunday, right? And I hope you do well on Assignment 1, and Assignment 2 is coming.
Bye, thank you.
And then, the third session, you know.