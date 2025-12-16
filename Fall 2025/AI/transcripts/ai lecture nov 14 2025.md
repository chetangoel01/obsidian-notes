It's a nice spot for an hour.
Let's see.
I'm gonna ask.
TikTok.
development, they said.
So I wanted to provide… we already started, is deploying, and to pass on the page, the thing. We are very thin today, maybe…
people will hopefully join. But I wanted to start the, presentation on the…
Basket Hube, an unfortunate name, I should mention, but the better name is AI Game Analyzer. So, as you know, the
the… around, I started, like, 2 years ago, a lot of people are building, everything under this kind of AI umbrella, so I said, why not?
And, I have a student who's actually building something like that.
on the side, and I kind of,
I borrowed kids inside to create this kind of a sign, which I hope you find useful.
Alright, so obviously, as the name kind of explains, we are going to be building something that can respond to, queries in natural language.
And that, based on YouTube videos of, games. Or, any, any recording of sports whatsoever, that would be a good thing.
Obviously, the first task in this kind of thing
I should say that this project, it's not a project, but anyway, this assignment is way deeper than these two tasks, obviously, for obvious reasons, and a lot of other people outside of academia are working on, because, you know, the people, everyone, everything that there is some kind of,
Dirty money involved.
People tend to aggregate there, right? And the dirty money is betting. Okay, so a lot of people are…
building this kind of stuff, because they want to, get rich quick, which is obviously why we're here. Alright, so, so obviously this,
We're not gonna start from scratch. Our aim is really to look at the AI kind of portion. Not that computer vision is not the AI, but, definitely
We are building something on top of computer vision. That's a semantic segmentation task. You can see that this semantic segmentation, however, is pretty good.
Remember I was telling you something, I'm not too sure if it was in this class, hold on a second.
Or one with mine.
Awesome.
Remember I was telling you that,
A re-identification problem. Remember I was telling you the re-identification problem, which is, kind of an important problem, where if I move outside of the point of view of the camera, when I come back, I need to be… I should not be double-counted, right? So this means that there should be some centralized system that keeps my vectors around.
keeps track of my vectors, is able to reacquire me and account me as the same person. That is called re-identification problem, and
And it's a little bit straightforward extension, based on what we have done, using a loss called triplet loss.
the, problem does not exist here. So, even if the camera zooms
way out, not really way out, because the way you can assume it's in the constrain out of the space, but the player goes out of the field of view and comes back, the player is uniquely identified. And you probably can guess why. It's not that we are doing anything intelligent.
But these players have a number, so obviously we're doing OCR and keep track of who player is, right? So that's the intelligent companies.
On top of that, Contrary to…
Atari Games, another kind of game-oriented kind of AI studies, which is also valuable.
The notebook over here is giving you
You can do something. It's a long notebook.
It's giving you something I wanted to refer to.
Any good stuff.
Okay, proceed here.
Where is it?
Where is the state of the game?
Okay, here it is.
It is also giving you this.
Allowing you to say, okay, who…
What are the positions of the triggers, right? When, for example, an action happens?
And obviously, the player who took an action can also be
sort of, obtained out of this state, right?
So the first kind of task… so remember these two pictures. So the first task, Nope.
The first task is to… Well… I need to pull back.
Is to analyze the player performance from comments.
In, professional basketball, NBA, the, you know, there are people, obviously these games are…
comment at this problem. I think two. I think there are two, people who are…
Essentially, what is going on. And, based on only the text information.
not video processing, based on only text information. I want you to try to, answer this query.
analyze that layer that scored the most in this game. Now, this game is, like, one and a half hours long, or something like that. It's a long video. I'm not really expecting you, to process everything, because,
In, at some point.
For text, it's very straightforward across everything, but for the second kind of task, you may want to limit yourself to some kind of minutes. If you…
if you limit yourself to one minute, it's no good. If you limit yourself to 10 minutes, it's okay, despite the fact that the game is, like, one and a half hours, right? So depending on your machine, make some decisions so that it doesn't, you know, you get a result by the deadline, okay?
One notable thing of this thing is that over here, the timestamp, right, has, is quoted, and,
So, this means that the system is able to ground
The response to specific events that the commentary has referred to.
For specific time.
So that's the second most important thing. First, to get something like that, where definitely the player who scored the most in this game is identified.
And in some instances, you can say, I do not know.
you know, who was really the… but up to this moment in time, there was this comment by this guy who said that the score was that, and this specific player hit the most of the…
He was responsible for one of the score, and the score once.
Over here.
Okay, so text. Now, in many videos, we don't have transcription, and I don't think in this video there is transcription, so what we need to do is to transcribe it yourselves, to obtain the text, right? So, ensure that you are understanding how to transcribe from,
exactly audio, and how to, from audio, from the video, and how to… FMP, for example, is a great tool to do so. And how to take the audio, pass it through Whisper, or any other model.
As you wish.
And,
get the text out, okay? You're not going to be, judged on how you did the transcription, okay? But what you will be judged on is the… the two requirements here, is to respond to with, some kind of a justification.
you know, of analysis. I'm not even expecting to see all of this kind of stuff, right? But definitely some analysis, is, is required, and, grounded to the specific times that,
That, the specific sentences were based on, right?
Any questions on task number one? I think it's a straightforward kind of task.
And, let's move on if there are more questions for dash number 2.
So that's number two is the most, kind of, difficult.
What I'm expecting, first of all, to happen is the so-called chunking, right? Chunking is a general term that we use also in other systems, like retrieval augmented generation and others, and basically it means
Take the whole game and separate it into the plays. A game is a very new place, and there are two types of plays, offensive plays and defensive plays.
So when one team obviously attacks, has the ball, and tries to score, it's an offensive play, and then exactly the threat is happening. So try to find a way, without any manual. Your call has to do the segmentation.
And again, you don't need to process a cold video, you can process just, let's say, 10 minutes of the video, right? But you do show that this thing, your code, is able to segment in place, right, the video.
Now, what's happened? That's basically that's number one.
And task number two, is,
within the place that, this, player is, active, like, within the place. Now.
I should tell you that, obviously, not all players are playing the whole duration of the game, right? So, players are going to the bench and coming back, or whatever, right? So, try to select a portion of the video where the player of interest you
you have a choice of what the player of interest is, it's your choice. It's, visible. I mean, it's doing actions, right?
Yeah, and obviously there are categories of actions, 7, I think. 6, 6 categories of actions that I'm actually, mentioning here.
And, for its identified action, so there is an action identification problem there, right, within the play, which lasts for… NBA games are very fast-moving, right, fast-pacing, so they last for a minute, I think, or some… something… some crazy number like this.
Okay, so for each, sort of, play, go and identify.
The action that a specific player took.
Okay, timestamp it.
So that's the first problem. What player, over here I mentioned two players, but, there's only one name here, because I'm asking you specifically for this player. One player.
And, what actually do?
And, provide, when they took the action, the bird's eye view
The state of the game, when the action was taken.
Are we afforded? What's going on? Okay.
No.
In this assignment, you are free to choose any model UEs, except…
from going to OpenAI API, create a key, or to any other API provider, right? You can use any model you want.
We are, very much aware that, of, limitations in terms of hardware that you may have.
You really need to document, potentially, these limitations that did not allow you to demonstrate their best performance.
And, submit whatever you have by the deadline.
Okay. I want you to submit it before Thanksgiving, because I don't want
you… I want you to spend a Thanksgiving break
relaxed, and in good spirits, and enjoy family time while you're doing your project. Right, so that's basically the reason. The 23rd of November has to be the…
a deadline.
All right, any questions on this, trivial task?
Yes. I just said, do not use the image.
Do not use an API.
Okay, let's, move on.
Pretty much, we do.
How's your project going?
Halfway. Halfway? Okay.
one person is halfway, the remaining… I want to just get a distribution.
25%. Who's 25%?
Who is 0%?
Goodbye.
Alright, so… oh.
With you.
So I finished the Transformers, and I owe you…
Something which is associated with transformers, but for the computer vision domain.
We're not going to spend a lot of time on it, or drawing extensively, because the architecture is the same, so let me just do the review.
Yeah, then I can, I can cover the vision transformers, or VATs at the section.
So, I think last week, we, I think this was the…
Last week, we started… remember, the journey started from, contextual three representations. Now, while I was going through this kind of lectures.
I mentioned the way that we construct them using this kind of linear projection, right, and the auto-important-like structure. And then I said to you, look, this is what is being delivered to the transformer. Well, the transformer itself may have an embedding layer, which is… itself learns, right?
But that output of this embedding layer, whatever it is, right, is context-free.
Okay? So don't take it that only…
Only outside of the transformer we can generate these kind of vectors. We can generate these vectors even within the transformer, but it wouldn't be context-free, because it's just basically an embedding, simple embedding label.
So we started this discussion as to,
why we may potentially want to take this input, which is coming in parallel now, we will talk about positional embeddings a little bit later, and
Take the… take this kind of inputs and project them in a new coordinate system.
So, if I ask you the question, why, what would you say?
Because there's different contexts.
Not entirely, in the sense that, I kind of, inside this kind of X, I have, mind you.
Definitely, I have this kind of context, right, the capital T, right? And
I'm saying that.
So, the capital T here.
I should mention the following. In this kind of treatment, at least, right, the capital tree is free, right? Or some kind of very, very small number, depends on the sentence. So I was kind of presenting
three snapshots of this context, right, where bears could end up being quartered. Three different snapshots in this kind of context. So.
So, it is, the projection I told you is kind of, required, because I want to, remember
What was happening with World Quebec?
Right? Just a linear projection was creating certain things for us.
Right? It was… the output of this linear projection had certain properties, okay? And what was the property? It's that, all of these kind of things end up in the same neighborhood if they mean the same. So something like dot product was actually,
And the end result of this is some kind of dual-product maximization for thekins that carry similar meaning. And also, we had also some kind of manifestation of analogies.
I didn't mention analogies in the discussion that we had, but it was… it is in your coreside, right? In the sense that we took an analogy, like, what a woman is to a man is, what a queen is to a king, right? And this analogy ended up being manifested via vector.
Right. So, NW Mavericks is all it takes to
completely change the coordinate system. And I think, if I remember correctly, just to emphasize a little bit this kind of point.
I think I mentioned, this is, like, almost like a parenthesis. I mentioned that, if I have, let's say, a coordinate system, let's say, with X1 and X2,
And I have, some kind of blue points over here.
Right? And I'm asking you
to do dimensionality reduction, always. The job of dimensionality reduction is to find a new coordinate system that you will… you will project this data.
That was the plane of PCA.
What is this coordinate system? The coordinate system, probably.
It will have two coordinates, obviously.
These two axes would be the best coordinate system you can Select out all the…
many, many possible coordinate systems, right? So, obviously, a point here has different coordinates compared to this coordinates to this, right? And this kind of coordinates may carry some
Meaning?
So, I kind of went and…
characterize these axes in the presentation of transformers, okay? I call this axis the objectness axis, I call this axis the verbness axis, and so on and so on, right? So.
projection, or this W multiplication, with specific matrices, which are going to be learned, the WQ, WK, and the WB, effectively end up creating this new coordinates.
Any questions on this? That's a very important point. Yes? All right, so, coming back… To this, discussion.
So in this kind of new coordinate system.
I'm expecting, after training, to see a behavior where a query is going to be responded with keys that can… are best appropriate to extract its meaning.
Okay.
Okay, so that was basically the whole purpose of this projection. So the WK and WQ are determined based on that. It's a retrieval mechanism. I don't want to call it a retrieval, it's identification of who can help me.
And then, the value, the value, over here is, was needed because at the end of the day, there's kind of a… I want to have also control.
on, how my, representation is changed from this kind of projection, accounting for another coordinate. I want to change the coordinate system also in the presentation of
of my meaning. So my meaning initially is inside the X, right, because it's a context tree of heading, and with the WV, I'm attempting, really, to take it to some other point, right?
So…
That point is V, it's not V hat. So, just to give you an analogy, imagine that you are going,
into the cafeteria that the professors are having lunch, right? And, you raise your hand.
issuing a query, right? Suggesting, hey, I need help in this math problem, okay? So out of the many professors, some of them will come to you to assist you, right?
So, this is basically the keys, right, that coming. These keys are identities, so these specific professors are going to be factored, right?
It's almost as saying that before you are arriving to the room, before you receive this assistance, your knowledge was filled.
Okay? After you receive your assistance, your knowledge, Seasonic.
You get out of the room with…
Copeled the different knowledge that you walk in.
Okay, do we follow this analogy?
Alright, so we built a nice little diagram.
That, demonstrates this, thing. And, finally, we mentioned that, We have a need for…
a multiplicity of this action, right, across, potentially, I will call it, even, okay.
It's not entirely true what I'm going to say, but sometimes, you know, when we're looking at text, sometimes we're looking at a certain grammatical pattern, again.
And then, at the same time, this specific, sort of head can extract significant, sort of, you know, identify this kind of pattern, and therefore adjust the values V hats. And for another grammatical pattern, another head will do it.
I connected this to the University of Builders and CNNs.
Where they are trying to find different spatial patterns. And over here, we are trying to find different temporal patterns.
Another way to actually associate with that, and I was thinking about this, is to think about that, in language.
We may have some form of, I will call it, yeah.
It's not entirely specialization, but, you know,
You can imagine one head to be a little bit extracting the comments of the source code.
the partners that you meet when you go to your Python code and extract the comments. And another kind of had to be a little bit more around the Python keywords, right, and be able to extract the meanings in this specific part of the intake, right? So think about this way as well. It's not entirely true.
Because of a technological mixture of experts, which I'll explain also today. But, it is close enough to be okay for us to associate a head to a different type of
Input. Comments versus code.
Yes.
So in the block diagram, when you have the final BI,
Does that mean anything, or is that… What is this error over here? I don't even recognize my own… oh, that is not the one decline. That is a circled X.
Is that what you're asking? Yeah, like… That's an X. Okay. That's a… that's a tens of X.
Let me just delete this circle, because it looks like a multiplier in some sort.
Okay, that's X.
That comes at the input, going to three heads at the same time, yes. Is it also going to the final B hat, or is it… No, no, no, this is another attempt to enclose this, to call it multi-head self-attention.
Obviously, we took care of the fact that we do not want to increase the complexity in that kind of stage.
No.
someone, I can't remember the name now, but, this past week.
Someone posted a 16-page analysis of
What is going on inside one single layer of the ashwater, which is basically this.
With the layer normalizations, as we discussed. And… Which we recovered, plus… The MLB?
Which has a nonlinearity.
And, if I find it, I'll send it to you. It's a lot of math there, but there was also a video who analyzed this kind of paper, out there, and actually extracted some value out of it. You can actually, so, over here, we did a lot of,
Hand waving, blow diagramming, analogy kind of discussion, but obviously the discussion is quite deeper than what we already did.
We are stacking, obviously, many of these, transformer layers kind of together. This end could be pretty large, depends on the architecture.
And finally, the head.
is, feeding to, soft marks.
After the features delivered by the body, to produce my workout, okay? This soft max obviously operates over 100,000 dimensions, or whatever my vocabulary size is.
Okay, now this is basically my Wi-Fi.
And then we spent some time on position embeddings.
And these position and endings were, treated here.
I know there is also rotational types of positional beddings, and a lot of other kind of categories. I just, in my notes, had this one, which is also the initial kind of positional bedding.
form, and I suggested that, this, number as to what is coming one after another, let's say, imagine 1, 2, 3, 4, 5, instead of appending into the X.
And, obviously, occupying, more number of beats there. You cannot eat.
Okay? Obviously, you won't… you won't be adding the integer now. You will be adding some… Vector.
that I call the positional and coordinator vector, which is formed using these strange equations.
Right.
which I plotted, initially, and justified
By looking at the digital equivalent of this.
Garrett's, what are these senior suites?
And finally, it suggested that we don't do that, we do this.
Like, an analog equivalent, effectively, where all the numbers are between minus 1 and plus 1.
And, for each location, we are actually picking up the values here, and this is basically my RI.
And obviously, this is the… access of all possible positions that I can have in my context.
That is where I think we stopped.
And… huh.
I told you that we would discuss about,
Mixture. We did have some discussion about a mixture of experts, And, basically,
I didn't tell you, though, Effectively, where the nature of experts
is, is implemented, okay? So what is this mixture of experts? For example, what we have said about unsample learning, where we have this, the time we have weak predictors.
And now these wheat predictors are going to be, sort of, different.
fit-forward kind of, sub-network, some might call it, where a token comes in, and it's being,
sort of, routed to these experts, right? So there are plenty of equations. I think the best diagrams that I have seen, I'm going to link them to your site when I have time, but it is called DeepSeq MOE. That's the paper. If you Google DeepSeek MOE, you can actually see the
diaggers, which I did not really include here. What I did try to include is, the, some form of analysis
With respect to what would be the best case and worst case situation that we have, right?
So, we did this analysis at a mean square error. This analysis is not quoted here. I think I told you that I'm going to do that today, just to see a little bit what's going on. And then, we can also do an analysis at a cross-entropy.
And Syria, that would still never happened.
That's right, that has never happened. So, it will happen at some point. I don't know if it will happen until this course is done, but it will happen at some point.
Bye.
But the risk weather is plenty to actually extract this, kind of,
Sort of, analogy with, cross entropy, because we have here cross-entropy, and,
So, can I just present a little bit with you now the… Wow.
That means weather or situation and the form of this. So let me just delete this.
I'm gonna talk about vision transformers.
I wanted to do this anymore.
So, in that kind of linear error, sort of, so this is basically the next, this lecture, so this is lecture…
Are we in the 10th election year? Who knows? 10th? Yeah.
14, 11, 14… Fair enough.
Continually.
Can you read it.
on MOE.
Well, this is, basically the situation in the… in that kind of, book. So, we had,
If you remember, some kind of, a sample.
of k predictors. So this is basically my, F of X is equal 1 over capital K. Summation from i is equal to 1 to capital K.
of Fi of X.
And, if the target is Y, the arrow, so… If Y is the target, Obviously, the error.
Yes.
Epsilon.
I is equal… FI of X.
Plus and minus Y.
And, so, in fact.
Let me… let me, put a… of X there, because obviously the label is associated with an accent, and so it kind of makes some sense to write it this way.
So what… so this basically is the error, of the IF predictor, okay? The error of the IF predictor.
Farrow.
of the highest predictors.
Okay.
This application is… okay. Anyway, so, what else? Okay, so the, so now we are looking at the unsample.
Stop.
mean square error of the holding, the committee also, I just know. And this is basically an expectation.
All right.
This is now F of X, not a phi of X.
minus… Y of X.
Squared.
That is the definition of misquared error, when the committee now produces a prediction, right?
This can be shown, that is expectation of 1 over capital K.
Summation.
over I of epsilon i, Hello.
of X squared.
Sorry, I need to…
summation of action on i, so the whole thing, the whole thing has to be square.
Which is a trivial kind of substitution over there. And after expansion.
we get…
that this is equal to 1 over capital K.
a quantity that I would be calling Me too.
It's a… it's a grip letter, so… but it's very close to… it's called U, but it's very close to the V. Okay, so if I call it V, it doesn't really matter. Okay, plus…
I'll explain what V is. 1 minus 1 over capital J,
Another letter which is being used here is C. Okay, so what is V? V is the expectation over
epsilon I.
squares.
This is the… Average.
Variant.
Off?
That.
Individual?
predictive.
C, on the other hand, Cheers.
the epsilon I, epsilon J,
that… which, for I, different than J, This is basically the average.
Or… covariance.
covariance between… Arrows.
So, there are two extremes in this…
error of the community, right? What is the first extreme? The first extreme is that all models are making identical errors.
Right? Which means… because the errors are perfectly correlated, right, this thing is going away. So, we have…
The error shows we have virtually two extremes.
So one extreme, is that, the,
This is basically the worst-case situation.
All members of the committee make exactly the same mistake.
And therefore, correlations are perfect in terms of mistakes, and therefore.
I'm expecting to get V as the mean square error of the chameleon, right?
Of the unsampled.
There's no improvement.
It's just like… Like, having… One…
Expert, or predictor, or whatever you want to call it.
In the case where this is the best-case scenario.
When?
the C is equal to 0.
Which means that They are making uncorrelated mistakes.
Making uncorrelated mistakes. So when the C is equal to zero, what is going to happen?
The mean squared error of the unsample.
is equal to V divided by capital K.
In other words, the error.
Yes.
decreasing.
linearly.
With a… with a… with the experts.
Which makes some kind of intuitive sense.
The linear is not…
Definitely, I mean, this basically comes out of the math, right? But definitely you would expect to get gains out of the committee members, but what happens in practice?
Is that, obviously the… this, require… this condition of, uncorrelated mistake is not, never met, because the degree of direction you can have in the system is not infinite.
So you can actually have, if you split the data, split, let's say, tokens around, right, obviously, for a finite calculation of tokens, right, you will actually see repetitive tokens being fed to the different amount of experts, and therefore you will have
some kind of form of dependencies, the correlations are introduced by the… and the same thing happens in sample methods as well. I'm not suggesting that this is basically the strict
I'll call it treatment, or what's happening in the,
experts in the mixture of experts, because we do not account for different routers' policies, right? A different policy could be a sparse policy, where I'm selecting 1 out of k experts, or 2 out of k experts, or I'm… I'm collecting all
The expert's kind of, view of, these, inputs, of this kind of tokens.
And so, depending on the policy, there are other equations that need to be written, but I hope
You've got the big picture here.
All right, so, again, the… I will mention the paper is deep.
Seek?
M-O-E.
For a block diagram.
and equations.
on… What?
Needs… We change.
In the baseline performer.
Which also comes in the… with the term dense, sometimes you're going to hear dense transformation. Could you explain, again, what are the conditions for best and worst cases? Like, when do they happen?
and dependencies, kind of… Sorry, the best… when do we get the best case, and when do we get the worst case? When they make completely uncorrelated mistakes, we get the best case.
Right.
The worst case situation is to have an ensemble where everyone tells you at the same time the same thing.
So either you have K, or you have 1, it's exactly the same. So when we, when everyone is making the same mistake, that is the worst case, but… Worst case, yes. So wouldn't that be when we get C equals 0?
No, no, this… And correlatingness of errors is equal to zero. Okay.
A correlates of mistakes, really.
Which, of course, brings us to the last thing that we need to discuss a little bit. What is the shape and form of the baseline transformer when the input is not natural language, but it is images?
Right?
And I don't have any notes here. Actually, it's very difficult to develop some kind of notes with images and throw things over there. But, I will, I will show you some, pictures and some graphs or block diagrams out of the computer vision book, which I'm using in my course, right?
So, in terms of visual transformers.
This book, luckily, is free of charge.
However, there is a catch.
foundations.
of computer vision.
And it's written by… professors at MIT.
It's a book that contains a good mixture of classical computer vision and Computer vision with deep learning.
Except on the very, very advanced topics that are… obviously, no one is writing books because they're busy writing papers.
And so the… so what is the cuts? And, I would like to take the opportunity to invite anyone who doesn't have anything better to do during the winter break. This book does not have
the Python code.
the graphs that are being there have no Python code. They have kept it somehow a secret.
So, sometimes people do exactly the reverse. They release the code, keep the text behind a paywall, disguised, release the text and the whole book, but kept the code close to their chest.
Okay, so one… if you want to teach out of this book, which is basically my job, is you… I have a tremendously difficult time pointing to implementations of the diagrams, okay? So if anyone wants to spend some time on it, please contact me after the course is done.
Okay, Foundations for Water Vision. Okay, so, out of that book, I… Where is the book?
Here's the hook.
out of this, I will go… I will take you to the transformer section.
you know, there's Chapter 26,
And, the discussion now is, what is tokenization?
what is the tokenization in the… in this kind of setting? Well, the tokenization is kind of a…
trivial here. We have an image. We will split that into some kind of grid.
Maybe 16x16, or something like that.
Some… some number which is comparison image, and if the pixels are not, in the grammaticals of this kind of number, you can resize it, doesn't really matter. And, as you can see here, these are basically your topics.
something strange… it's a strange quotation. There's no intuition, behind it. Well, there is some intuition behind it, but I personally kind of find it strange that,
Either you line up the tokens like this.
or you somehow involve the country in the tokenization, the specific positional, information. I think, you know, this, head behind, you know, below the head, there is this neck of the…
of, the giraffe, right? All of that is not, important, right? So either you
sort of, do not obey any, sort of, positioning information, and… or you be compliant in a special kind of, layout of the image, the performance is, one and the same. This is basically what I… why do you need a pilot and call to verify all that.
But nevertheless, that's basically what is claimed, and to some degree, there is some kind of truth to it, which I'll refer to it a bit later.
Out of every batch, you will read now your token.
buy a featurizer.
Okay.
The refrigeration could also be the raw data, which are, present in the image.
That's basically what you're talking about.
And of course, we're going to have a capital X matrix, again, to pack all these kind of, tokens, and exactly the same architecture is going to, to be,
Attended with the attention kind of mechanism.
And in this kind of attention kind of mechanism, Obviously, the…
I want to show you something which is probably not this picture, I want to show you something which is probably the best picture of this version, actually, if I do it.
I wanted to show you this.
So, now the queries and the keys are a little bit more, straightforward to understand.
Like, over here, is my queen is vector.
is associated, let's say, with, color. In fact, the multiple head is also very intuitive. One head could address color patterns, another head could address
texture patterns, and the other head could address shape patterns, right? So the multiplicity of heads is kind of very straightforward in computer vision, right? Much more than what I was explaining.
So here, for example, you have, three things here which are the queries.
And you plot the capital A matrix.
Right? Which basically says, okay, in this kind of image, if you put everything together, in this kind of image kind of height and width.
What is… are you going to observe for this week?
What tokens, in other words, you would, expect to receive count from, right, for this quid?
This is color. The color here is this, right?
So… You notice that Dawkins…
that are similar to this color are responding. Do you see them? A white means closer to one attention, and a black means zero attention.
For this specific, point, right?
You can actually see that, this specific point is the blue one, right? And it's actually the…
this thing over here, right? So you can see that the tokens that are the members of the
Antelope with this, I have no idea what the animal is this, are assisting to extract the meaning of their main
portion of the envelope, which is basically my query. This is my query here.
Aren't we following?
So this basically means that a lot of ones, or close to ones, are around here.
And, maybe… Something over here says, hey, maybe I can help you too.
Because you probably noticed that this color here is very similar to this color.
And of course, you have something on the ground.
And as you can see here, a lot of photons over here are lighting up.
When there is similar colors in the picture, and that is obviously ground.
Other than that, the equations are, identical.
Okay, the relationships are identical, block diameter, okay. You said that patches resemble tokens, like we saw in the original transformers. Yes. So, is the atten… when we're talking about attention, is one value of attention given to one patch? Correct, to a token, yes.
So, that will mean that if the path size is pretty big, then there would be very few, yeah, of course. As I said, the typical number, and this was the number in the original paper, is for a…
I can't remember now. It was 224 or something like that, by 224 kind of image, or maybe it was a large number, it was 16 by 16, the token, so it's 16 by 16 is a pretty large number.
That's why you see them here as being our smallest players.
Okay, so that's basically, so every time that you have your CSV ID, you remember, I think it's exactly identical, but the tokens are now, the ones that are identified.
Okay, so this is basically as far as Visions Maspoon is concerned, and the chapter here
If I remember right, it's Dr. Walsh, what?
26.
Okay.
No.
In this kind of lecture, basically, this was a wrap-up of transformer neural architectures, and here is what is going to happen
Over the next, kind of few weeks.
So… I am going to switch to… reasoning. Now, Reasoning is going to… come into… parts.
One part is underdeveloped, or not developed at all.
But the part that I'm going to talk today is developed, and it's called logical reason. And it talks a little bit about, writes you about a little bit of,
your, search algorithms and stuff you have done in high school, right? And, your, this kind of, methods of, what is called discrete math. Anyone has taken a discrete math course here? Some people do, okay? So, you will find the description kind of at home.
We are here in this part of the course, this logical reasoning looks and feels like an island, okay? Almost like a no-man's land that no one visited, kind of thing, right? And the reason why it looks like this is that,
The second part, which is bringing together neural computation and symbolic reasoning.
has not been developed. So this is called neurosymbolic reasoning, and it's a very…
a sizable part of modern AI today. So, with that kind of caveat, at least I hope that some of you may have some kind of a tooling here from this kind of discussion to go ahead and understand what is a
propositional, what is first-order logic, and things like that, right? So here, I'm going to focus on propositional kind of logic parameter.
After we're done with the logical kind of reasoning, we will be dealing with classical planning.
Which is largely means for us,
The, sort of, presentation of,
how to solve classical planning kind of problems with a language called PTTL, okay? Another kind of dry kind of topic without a lot of concepts behind it, however.
the literacy is obviously kind of very vast, so I will be focusing after classical planning to,
planning with interactions. This means, then we go
To this kind of, approach, where marked additional projects are going to be discussed.
And, reinforcement journey.
The reinforcement learning
After we cover that, normally, in a more advanced course on AI, someone could take this course and develop a more advanced course where enforcement learning improves.
They're built on top of where we left it with Transformers just now.
The next token prediction.
So reinforcement learning is going to be the tool that this next token prediction will become what we are using every day in ChatGPT.
But we don't have the time to, cover, that part, that part, right? So, but again, something to think about after the course.
Okay, so I'm gonna just start this kind of a logical reasoning.
In the topic of logical reasoning.
I want to, make sure that we have,
we understand that this, kind of technology, is not really, you know… Yes, definitely, people developed it first in the 70s, right? And this was basically what the AI was, right? But, recently.
Logicians have found, have experienced some form of a comeback.
AWS has, created a whole system, right, to, manage their…
For… to be able to answer with logic, the following question.
from customers, without necessarily them being involved, they just provide the machinery for this, which is called Ethereum pluming, where Query comes and says, okay, is this…
Silver.
Because, you know, when I'm doing infrastructure, I have potentially hundreds or thousands of servers, right, or millions of Docker containers.
Is this Docker container connected to the internet? That's a classic cybersecurity query.
Because most of the time, I don't want this to happen.
And so the, service, the thing that powers the responses to these, queries is, based on, logical reason. So, people have been, as I said, have, built applications, so that is not really, I guess, today's kind of treatment.
Just because we have to. We don't have to do logical reasoning, but I have it in part of the syllabus, so I'm gonna do it. Okay, so logical reasoning. So,
So, logical reasoning can be, sort of divided into, let's say, two parts, two parts of the discussion. The syntax.
Which, obviously, the language that I'm expressing.
Logical, sentences and the semantics.
But, effectively.
show how the… a sentence could be true or not. So, in terms of the syntax, let me just write it down, and how to represent the sentence.
Okay, and, with, semantics is,
How weak?
do inference.
Excellent.
Based on what we will see here, models.
Let me put this word… world models in Toronto.
As it turns out, inference can be done purely based on syntax itself, right? And this is actually called theorem proving.
So, I would say also here that how we make… How we do influence.
to inference.
using… See, index.
Without any modeling.
October?
the so-called Fiora.
proving.
Moving.
We are improving.
Okay, so let's look at the syntax first.
But we're gonna cover… we're gonna, cover a language.
That consist… that consists.
of propositional symbols.
That evaluated through or false.
and operators.
Which operators are.
Negation.
contraction.
This junction.
implication?
And,
double implication.
Let's start with a trivial kind of sentence, and I'll show you now a model.
that, satisfies model of the word, in other words, satisfies the sentence. Here, the word is, the…
group of, my propositional kind of symbols, and, you know, the, let's say, assembled, let's say, A1,
It's, let's say, rain.
And?
Wet?
And, obviously, Since these are propositional kind of variables, but A takes the value of…
0 or 1, let me call it like this, and W, the wet base value 0 or 1, and obviously.
This is the… model.
Of the world.
Not… satisfies.
the sentence.
And we have some kind of notation to call this model of the words inside the sentence, and we'll call this
M… Oh, fuck. A1?
So, when you see M, it's one model.
However, I want to give you this kind of a sentence, A2.
as Arain, Arsenal, as you can see over here, there are multiple models, let me call them M1,
M2 and M3.
that satisfy this sentence. And this multiplicity of models, we are going to call it with capital M, A2.
M1, M2, M3.
So a model satisfies the sentence, When?
It evaluates that sentence to true.
All the prompt sentences are going to be stored somewhere.
And this summer is called a knowledge base.
So we are going to be, referring to that kind of a knowledge base, because as an agent kind of experiences the environment, they will see an agent like this soon.
we are going to be storing.
Bruce?
That, we are… obviously corresponds with sentences that govern the…
Action of the agent, let's say a game of some sort, and perception information, all in the database that we call knowledge base. In logic, we call this knowledge base.
So…
This, kind of, obviously the sentences are governed by, as you can see here, the trivial ones, from, these results were taken out of
proof tables, or… and I think one, rule, one kind of operator over there is kind of a little bit more,
It's kind of more difficult to understand, the truth table,
And, so I'm gonna just, remind everyone about the implication rule.
But it's also going to be involved in, influence to some degree.
So, the implication rule is… Cute.
The propagina symbol is P.
Q.
B implies to… And P, W indication Q.
It is FF… This is true.
And… This is F.
FD?
This is true.
And this is F.
and through F.
Now, this is F and F.
And true, true, this is true, and this is true.
There is a kind of a way to replace this implication rule with other operators that involve some kind of conjunction.
And therefore, obtain these values, okay? And I'm going to write a conjunction here as… B.
live skew.
is effectively
Nope.
Bullet.
B?
And… Not fuels.
Cheers, nope.
B, and… not viewed.
Which is equivalent, naught.
be poor.
fuel.
If you…
Obviously, we don't understand how this, sort of, equivalences work effectively obtained, because that's basically a logical kind of rules, or that we call equivalence kind of rules. And I don't expect you to…
go down to this kind of level. What I expect you, however, to be able to do is to consult an equivalence loop, equivalence table, that is, that has, it's called it in Chapter 7 of your textbook, and it's called, the table is, I remember it always because it's also a store, 7-Eleven.
Okay, so the 7-Eleven table, you have to have a copy of, or I will provide a copy in the final exam, but just in case I don't bring a copy of this 7-Eleven table with you. Okay, I'm going to show you. Unfortunately, the book is, like, 2,000 pages.
So, I don't carry it with me, so I will attempt to show you the, that kind of table to understand what's going on.
In a moment. But obviously, there's not a lot of intuition of this kind of implication, so maybe a kind of a translation of this rule to some kind of English sentence may help us kind of understand what is the… why the false implies false
is true. Okay.
I'm… I divided… I'm not so sure how successful this analysis would be.
But I divided this tool table into this kind of four cases, and I'm writing.
If it is… not raining.
It… ish.
not.
Cloudy.
And, my intuitions suggest that this is true.
Second… Eve.
It is not.
Rainy?
It can't be.
Cloudy?
And this is also true.
I hope you're following to some degree, right? It's not the perfect kind of analogy, but it knows enough.
If 3 is if… It is raining.
Eat.
cannot… be cloudy.
Oh, whatnot.
Well, I mentioned the is cloudy in the first sentence and can in the second sentence, so…
Let me just replace this with… forget about Khan is.
loudly here on the second sentence. And forget about this, I have it in my notes, that's why I've spoken it.
It is not cloudy.
That's bullshit.
If it is raining?
Thank you.
No, again, it's… it is lousy.
And this is, true.
So I tried with this kind of English sentences just to sort of provide some kind of intuition about the truth table on this kind of implication.
And obviously, the truth table of the double implication is, again, this equivalence kind of table comes into perspective, so this double implication is P implies Q, and Q implies P.
And if you have the truth table of any of these two implications, obviously you can do… write the truth table or the double implication.
Which are the entries which I have highlighted.
This inspires the truth table is important. I don't think any other operator, has some kind of difficulty understanding the truth table except from this implication.
Yes, go ahead.
from this definition, for the… for PW, there is Q.
Should the first entry be true, then?
What's green.
B, government place youth when voter forms.
Shouldn't that be cool?
You mean the double implications rule? Yeah.
Yes, there is a typo here. This is true. This is true. Yes, of course it is true.
Of course.
Yeah, yeah, good catch. I think someone else in the past mentioned there was a problem with the truth table in my notes, and I need to, a little extra.
Welcome.
Yep, that's start.
All right, so, so that's basically the truth table. What I would like to do now is to,
Go over,
A demonstration of the inference using the knowledge base.
As in the form of a kind of a small game from the 70s, called, google's work.
In the homeless world, there is an agent that is going to move in a 4x4 fictitious world, where there is going to be a monster in there, and the name of this kind of agent is to discover the gun, then get out, and permeate the game. That is basically the agent. In the process.
of executing the game, acting in the game. The agent is very conservative. It cannot move unless it is certain that the next cell it will go to is safe.
How do we check that an Excel is safe?
It can sense the world, it can sense.
we are receiving perception information, in other words, that somehow is correlated to what could actually occupy this kind of cell. Obviously, the monster is smelly.
So there is, this kind of a sense, but there are also some other senses that I will explain. It's a kind of a very small kind of game, but, explains the need for logical reasoning.
So first, we will do the logical reasoning using our own brain, and then we'll see how the machine will do logical reasoning that agrees with our brain.
Okay, that's basically the demonstration.
So, I'm gonna draw the English word now, so this is…
I'm not sure if in this critical map you already have seen this, or I…
Okay, so this is my 4x4. I'm going to use coordinates.
This is where my agent is.
B.
Yes, Breeze?
It's one of the senses.
What is out there, please? Okay, it's here.
you have here… Hover here… I'm sleeping off here.
And here.
S is stench.
This is where the monster is, so how do I draw a monster now?
I don't know, something like that.
Okay.
Alright, so this is where the monster is, and
The monster, apparently, is very smelly, so it kind of,
When you go to that kind of cell.
adjusting counselors, you can sense the S.
Another stench?
And, by the way, this is also where The gold is located.
Next to the monster, in other words.
Which is my aim, to grab it and exit the game, right? That's where the game terminates.
There are some kind of, beats.
Which are surrounded by breezes, right?
So the moment you fall into the pig, you die.
So, our first thing to, so basically that's what's gonna happen. So, let's see. I am going to, first of all, start populating the knowledge base, which is now has no rules, with a set of rules that are
Rules of the game, before even I start acting, that's just like in driving, before you start driving, obviously there are certain rules, right? Or the traffic rules that exist already for you. So I'm gonna write down these rules in sentences, and I'm gonna store them in the knowledge base.
One thing that will become very apparent is the more rules… I want you to remember these rules as constraints.
The more rules I add in the knowledge base, the more difficult it is for the knowledge base to be satisfied.
satisfied based on the definition that we just had, right? The knowledge base to result into proof, right?
There is some intuition about it, right? And we'll see the intuition with numerics in a moment. So I'm now starting the population of the knowledge base. I'm going to write the knowledge base over here.
Okay, KB is… knowledge base?
I'm going to call Roush.
instead of the sentences, the A, B, and so on, I'm just going to call it R1.
What does it say? It says that the game can't start without
terminating immediately. There's no PET in 1-1. There's no PID in 1-1.
R2.
311.
double implication, P12, or P21. I told you.
That, when the pit is, there, the adjacent cells, whenever it's possible to have an adjacent cell, it will… you can sense breeze.
Are you happy with this, second, sentence? Okay. Third sentence.
B, 2-1, double implication.
P11 org.
P.
Toto.
Or… 3, 1.
I hope the indices are correct.
Okay.
That is the secondary thing.
I forgot to mention, something also. Let me call this something R0.
not… W11.
Okay? There's no monster in one.
Where you start the game from.
And now, the agent… will take actions.
To move to a Jasmine cell, looking for this bolt, without getting killed.
Okay.
So, state.
is, I'm going to create a table where some columns here will be the state of the game.
percept.
perceptive information which I'm receiving.
inference that I'm making.
logical conclusions, in other words, inferences that I'm making.
And after the inference, Actions that I'm going to do.
I'm starting.
prompt.
the same variables, L, location.
And, another variable I wanted to reduce.
CITA, not with E, CETA for Orientation.
orientation.
So, theta E, agent, looks to the east, right? Theta E looks to the east, right, the agent. So, looks kind of that way.
And,
basically decides to, what are the candidate kind of cells you can go to? Either the cell 2, 1. Remember, I always quote the horizontal axis for the first number, and the vertical axis for the second number. So either it can go to 2-1, or it can go to 1, 2.
Right, these are the two cells that can go. Let's assume that,
Takes an action to move forward.
You start the game, in other words.
In the east direction. Then it, receives.
But surely, I forgot to mention that.
Being in a one-on-one already receives That, there is no…
with this, there's no breeze in 1-1. In terms of the per second information, I'm not receiving a breeze in 111. I'm not receiving stents, I'm not receiving, this gold is also sensory in the sense that the gold is,
glittering, right? So, and yeah, so these are basically my sensing kind of information.
Good.
So I can write now a whole bunch of rooms here.
Right? Because, about all the perceptions. I'm just gonna focus only on one here, just write only one, but you can, you get a point, I hope.
The action forward, right?
R6.
There is no pit.
Into one.
that I'm occupying, because…
I almost said it in the reverse kind of order. I'm receiving perceptions, I'm making inference, and then I'm taking the action, okay? I'm not moving until I'm making the inference. So I'm in cell 11, I'm receiving perceptive information.
that says everything is clear, okay? I am making the inference that since everything is clear here, in my adjustment cells, there is no P, right? Therefore, I can just go forward, okay? And I am now at timestamp number 2, I am at the location.
is equal to true, still facing east, okay?
And I'm receiving now the R5s.
rule, which is, there is a breeze. I'm receiving breeze when it's location to 1, right? Then I'm making the inference.
That.
there is a pit, right? I'm making this inference, that there is a pit.
in.
Where?
3, 1…
or P2, correct? Again, we… we are making the inference, right? The machine doesn't know how to make the inference here, but we can make the inference that there is a P in either 3, 1, right, over here, or…
in Tutu.
Correct?
Who do you want, is this going?
Or, yeah, or to 2, or I have to pick. In other words, there is… there must be a pick either here or here.
Because I'm receiving breaks here.
So… My action… is,
that I have to go back.
My action is to go back, because I cannot move.
forward, right? I cannot go up.
And therefore, I will need to go back.
Sweet.
I am stuck now on L11 again.
And I'm now, because I went back, I'm facing west.
Definitely, I'm receiving, again, there is no B11, Obviously.
And, I'm making the inference, there's no pit in 1-2.
There's no P in 1, 2, and therefore I clear the cell P12, and therefore I can go up.
I am going to… to turn.
Aaron?
You're entered.
Definitely, and move forward.
Huh? Yeah, totally.
sometimes I'm actually compressing the actions.
Okay, so this is 3, this is 4.
So, L, I'm now in 1, 2, correct? I'm now in 1-2.
Annually facing north.
What sense am I getting? I'm getting S.
What S is that out of many? It's S12.
Right? S12 is true.
what difference can I make if S is true? That there is a monster in the adjacent cells. So…
Having said that, having said that.
I'm inclined to write, Jr is a monster, It's either here, Or here.
What?
However… We already know that there cannot be a monster here.
Why? Because in our previous experience.
We didn't get a stench here.
So that is a… it's trivial inference, right? But it's going to occupy us.
as a kind of a problem, how we make this thing as a machine, obviously, as I told you. So, out of the two possible sort of true values that could be there, and therefore I was almost ready to write this junction, I'm only writing one.
So I know now where the monster is.
Okay, and I also note that since I am are not receiving.
I'm not receiving any breeze here, then this cannot be a peak here.
That's a second inference which I'm making.
Therefore, I am going.
Oops.
This is Kutu.
Be careful. This is 2-2. P22.
Okay, then what I'm gonna do, obviously, I cannot go north, because that's where the monster is, I know it now, so I'm gonna turn, and then move forward, right? Turn…
And forward.
And then this will learn me.
Ian.
what is this? Think, L22?
Let me… I'm not gonna write anything here.
It's true.
So, I'm now in, L2 tool.
And, facing east.
Facing east, right?
I'm in… I hope you remember, I'm now in 10 to 2, I moved, right? I'm in 10 to 2, facing this way, right?
I can, go here and, waste our time, and, go here.
find myself in a situation where I'm not safe.
And therefore, I have to go back. I'm bypassing this possibility, right? Nothing… it's like a loop. I'm back where I started, right, in L22.
Out of the two choices to go east or go up, in other words, and I'm not gonna write, obviously, the go up.
But you can understand that I could have reached the same conclusion by going in some other route, right? But then I couldn't really move, because…
I cannot know if there's a bit here, or here, or, or… yeah.
So, we are going up, and then the game terminates, because we are sensing
So, effectively, what I'm actually writing over here is, if… Boeing.
East.
I have… Google.
To L22 again.
Right? And, the last… so there will be many lines, potentially, that this will last, this journey. And then finally, I will decide to,
moved well to 3.
Facing north, in other words.
Then I'm getting all the senses.
S23, I'm getting stents, I'm getting breeze.
But also, I'm getting the glitter from the gold.
The moment I get a glitter from the gold.
Hi, Grab.
The goal, and the game terminates.
Okay, it was a long, Shakespeare.
Explanation of a trivial kind of game, with, however, some, lessons learned in terms of
I'm building, gradually, a knowledge base. Remember, the knowledge base started with, just, four rules over here.
And as I was moving around.
Perceptive information stored in the known space.
And, inferences are stored in the knowledge base, so everything is stored in the knowledge base.
That's basically the… So now that we have done this, for example… This, kind of game.
I am going now to suggest
How we are going to treat this knowledge base, and this is basically the most,
Out of the core, discussion of today, in terms of logical reasoning, that is.
That our interaction with this knowledge base.
Again, the definition of KB is… A set of sentences.
that represent?
Rooms?
Very warm.
model.
Percepts.
inferences.
And as I told you, Rules of the world, model, passage and inferences. I want you to think, think.
All… We're almost… as constraints.
Logical reasoning will become more and more difficult as We have to go…
Potentially many sentences to make that inference.
That's the intuition.
So over here, we will adopt,
sort of to visualize what's happening. We'll adopt VIN diagrams.
to visualize And, understand.
the KB operations.
The scale operations are… The queries ask… And, the…
writes. This is basically the reads and writes,
But the reader does not read, basically, queries, right? And,
The rights we call, in logic, original terrace.
Fair.
These are the… these are the two variants, ask and tell. So…
Yeah, knowledge base is actually over here. We can actually imagine that We want to store.
a sentence into the knowledge base. We have done that.
Just now?
And the knowledge base may respond with 3 different responses.
I knew that.
This is what we will call entailment.
He can say that I don't believe that.
And this is a contradiction.
And… You can also say that
Updating.
That we call the… Contingency.
This means that this new center is now part of the knowledge base.
Any questions on this, tail operation?
Trivial stuff, okay? So this is basically, the second verb.
The second verb is ask.
Nope.
False.
Or… I don't know.
are the three responses. I'm asking the knowledge base to tell me if this thing is
if you have any information about this sentence, right? Is it true? Is it false? But also, in this logic, thing,
something which is not happening in our relational kind of databases, you can respond also, I do not know.
Okay, so what I want to do now is,
Because this thing is, effectively going to…
as you can see here, not only store rules, but also tell us whether something is true or false. Obviously, it's a very fundamental operation, so we need to understand all of these operations and their responses.
And I'm gonna start… with,
With a third.
practical creation.
Entertainment.
There's a reserved special symbol for entailment.
And, I'm gonna use it.
And, in, in,
In the left side and the right side, you may see sentences or knowledge base being involved.
And, it will read that… That…
effectively the entailment, which is basically the manifestation that I already knew that.
is, knowledge base and tells the sentence, right, that I already know, right? So that's basically the response for a knowledge base. So I'm not… I'm not going to ignore this sentence from you, because it's already part of my database, of my knowledge base.
So, as I said, we are going to use some kind of Venn diagrams.
And some kind of examples to understand entailment. So…
I defined earlier the capital M to be all the models of the world that satisfy a sentence. So I'm gonna just,
for this, like this, like an overall shape, like this. This is M.
Hopefully one.
Starting, I forgot to mention that.
Starting.
This is a starting position.
Kb.
contains sentences. A1, A2.
right now, KB has a sentence, and it doesn't matter what sentence it is, it does basically have these two rules inside, right?
And,
the Venn diagram is… In this case, it's this.
The set of models that defied just the sentence A2 and A1 are like this.
If both of them are in the knowledge base. Then, their intersection
Right? Their intersection, this shaded kind of area, represents what?
represents the… Set of models that satisfy the knowledge base.
Because to satisfy the knowledge base, You…
need to satisfy both of these sentences, right? So, this is… the section of AI.
Well, this is a… The intersection symbol, okay?
The unit is exactly the opposite.
So let's see now, and this is very kind of important, so I'll go slow.
Suddenly, another sentence shows up that I want to store in the knowledge base, okay? The knowledge base… the situation we're going to block now is that the knowledge bases already knew them.
So, I'm… I'm issuing the operations delve, KB, A3,
the Venn diagram that… We'll manifest an entailment.
will be. If this is the set of models that satisfy A3, And…
sarounds.
I just copied this thing over here.
sarounds.
And closes, in other words, right? The set of models that satisfy the knowledge base. The knowledge base will respond, I know that.
And this is a little bit,
A little bit, kind of difficult, so let me write down the equation.
In other words, if… In other words, what I'm saying, intercept of models satisfied knowledge base. Intersection.
interception with a set of models such as Phi A3.
Is equal to a set of models that satisfy the knowledge base.
In other words, the set of models… the presence of AE3 does not change the area
of the shaded area. Does not change the shaded area before that sentence A3 arrived, as you can see, it didn't.
The knowledge base says I already know that.
That's how you need to understand it with this event diagram.
Are you following?
Sure.
Then? We ride?
What its base entails.
A3.
I hope it's clear.
And let me write as a note.
Nope.
That… the… Shaded.
Any odds.
That… Set this fights.
the knowledge base.
He did not.
Change?
wrong?
There.
Presence.
of… Hey, sweetie.
Which, of course, points to another thing.
If it did, then some other response will be water, if it didn't. So let's see how this,
See if we have here.
Are we okay? Can we proceed?
So, the second thing… Which is natural now.
Is to… for the contingency.
So… This is a set of models that satisfies the knowledge base.
A3?
It attempted to be stored in the knowledge base, right?
If A3.
The set of models that satisfy the knowledge base.
It's like this.
And therefore, they knew.
In other words, the A3 shrunk
The area of the set of models that satisfy the knowledge base.
Then the notice base says, I didn't know that, I need to update it.
I need to accept you as a uniform Information.
Okay, that's the… that is the essence of this kind of end diagram. So… In this case.
the area.
with… That… Set of models.
that satisfy.
the knowledge base.
Represented by… M.
of KV.
is reduced.
Bye.
the… Sentence.
A3.
before, We have this coffin, And now… A3.
cost this reduction.
With the green area.
That is the new set of models satisfied the nominally, after the storage of acrylic.
Which, of course, brings down to the, easy Venn diagram, the contradiction.
Oh, I forgot to mention the equation for the contingency, so this is important.
So, what is the equation? M.
Oft.
KP, intersection M.
A3 is a subset of M?
David.
That is the… That is the reduction in the area, in kind of a geographic area.
reduction, the visual part of that. So, A3 that used a set of models to satisfy the knowledge base.
And, this thing, however, this thing, however, Then… None, Seth.
of models is a strict subset of the set of models that try the knowledge base. It basically says, yes, it reduced it.
And also, it does not contradict what is already in the knowledge base.
So, this guy is… the reduction.
And this guy… Yes.
what I know.
does not.
contradict.
A3.
Okay, now I wrote the equation. The contradiction now is straightforward.
the Venn diagram is… Trivial.
This is the contradiction.
no common ground, in other words, between the set of models that the… is known to the… satisfies the knowledge base, and the set of models that survives A3. That's a contradiction.
That's a… I respond, but I disagreement.
By the way, where in the previous example of the… of the humongous world.
We, met, this, contingency.
We met this contingency. Every time that we're making a logical inference.
That resulted into that new piece of information, either a perception or whatever.
To be stored in another class, right? So…
So this is basically the term of collections.
And they ask corporations.
Have their own kind of equations, but these things are a bit easier to kind of understand.
If I'm asking something about sentence A,
The first response, the response of true, mints…
Something which you were gonna tell me.
False.
Means something that you also want to tell me?
And… I do not know.
Which, of course, will be dependent
will be dependent, dependent, or effectively the item don't miss its…
It's neither true nor false. That's basically what it is, right? So, it's easy to… for the I do not know. What should I write as an equation
It has to do with entailment.
Because when the knowledge base responds that sentence A is true, when already it has information that makes this sentence true.
So, going back to the operations that could result into… I already know that, right? That's basically the equation, right? The true is coming from that. This means that
knowledge base, Knowledge space, entails.
Great.
again, go back to your entailment kind of thing, right? And see how…
The fact that the knowledge base responded that the sentence is true means that someone
You know, it's… it's already… not someone, the knowledge base contains that information.
So it's a… you can write this.
Obviously, we can write for the false Bruce.
What I don't know is we cannot unveil.
Either the positive, Either the sentence?
Or it's… Negation.
Okay.
So now that we have presented the operations now, and we know now how to, store things in the knowledge base, and how to do, with our brain, logical reasoning, now we need to study
just to…
methods, or where the machine could potentially do the logical reasoning for us, okay? So, the first method is…
Logical reasoning.
Weird.
Model checking.
So… I want you to think about the following problem.
I want to… You could tell me something about… this sentence.
February this has negation of P12.
And I'm giving you some conditions, that a knowledge base is in a specific state.
This is basically the problem.
Or… the ask… operation.
Sean?
Determined.
the response.
Of the knowledge base.
When?
Then on its base.
Is… at a specific timestamp, T is equal to 4.
What is this timestampus defaulted before? If you wrote what I wrote, you probably remember that when we were running the Umesh World, we had 1, 2, 3, 4, 5, 6, whatever, right? So, this means that we are going to have, inside the knowledge base all the rules we wrote, from R0
2?
R5.
Okay, so if you don't believe me, I have it in my notes, let me go back and say, and suggest what's going on in 4. I will do the same.
form.
I would consider these rules R1 to R5,
I could have far more rules, right, in the knowledge base, but then I won't be able to even present it as a table, okay? So, what I'm going to do, in other words, is I'm going to…
and this is basically the essence of the model checking, is… that's the… that's basically the assumption, that only R1
Ange?
only R1 through R5 rules.
are already… Stork.
In the knowledge base.
Let's make this assumption.
You go to this rules R1 to R5, and write down, and you're writing down, all the symbols involved in those rules. All the possible symbols involved in rules. You don't, you don't miss one. I'm writing down the symbols.
So, to answer the question.
with model checking.
Probably we guessed the method already. I am writing.
listing.
Meh.
propositional… symbols?
involved.
with… R1 to R5 rules.
This is… B11.
B11.
B… 2-1.
P12.
P2… 1.
P22… P31.
These are 7 symbols in total.
what are the possible values of these symbols? Obviously, true or false, right? I mean, each symbol could be either true or false, right? So we have two values, binary, these are binary symbols.
And we have 7 of them.
what I'm gonna do now is I am going…
to enumerate all possible models of the world that exist here with just these five rules. How many rows am I going to have?
binary symbols, and I have 7 of them.
Due to the power of… 7. I'm going to have 128 rows.
Sure.
I'm going to enumerate…
all models, Endline.
with I belonging to… 1, 2, 3…
Due to the power of several.
I have 128 models.
that I can form with these 7 symbols.
Any objections to that?
of 128 models. Each of these symbols
could take the value true or false, right? And therefore, I will go into my knowledge base now, and for this kind of rules, I'm going to evaluate
certain things. I'm gonna just,
you know, show you the table, okay? Now, I cannot really draw the table here, so I will try to find… I didn't bring it with me, I didn't do this search before, but I know that there is, in the internet, in the book that I was telling you about, so let me do the keywords here.
No, and, model checking.
So, in your textbook, you will see the… this table.
You will see this table right here. Come on.
By the way, this is a table 7-Eleven, which I told you to print and have it with you.
Okay? These are the logical equivalencies. We will be using this table next time we meet, because we need to cover the second method of doing logical reasoning, which is syntax only, right? That's the most difficult method, obviously.
And you need to have this table in front of you, so praise me that I'm bringing it the next time, and also for the final exam, if you are unfortunate enough to
creates a logical reasoning question, that is. Okay, so, let's see. Where is this table?
I'm, I'm, passing this algorithmic stuff.
that, always involve some form of,
forward search kind of algorithms, with backtracking, stuff like that. Come on.
Well, no, I skipped the table.
Mark, I hear this.
Here, this is the table.
I was looking for it. The table is divided into two parts, with a double vertical line separating the two parts.
On the left side, As you can see.
There are the symbols. Involved in the rules are 1 to R5.
These are the symbols we wrote.
One of the symbols, by the way, is involved in the query.
The query has something to do with the pit in location 2-1, right? I think it was negation.
Of P12, right?
Yeah, so…
We are then going, and for every room, R1 to R5, we evaluate the room to be true or false.
Based on the specific model of the world.
Every row has a different permutation of what is true, what is false. So, for some rows, R1 will be true, and for some other rows, R1 rule will be false.
So, we are going… And, do this thing, right?
When the knowledge base at this specific timestamp is
satisfied. When all of the rules It contains… are true.
So, it's a… it's a conjunction.
R1 has to be true, and R2 has to be true, and R3 has to be true, and so on and so on, right? This is why I was telling you that rules, as they come in, they introduce constraints. They make their life more difficult, right? Because they're adding more rows into this primitive machine called model checking.
Let's try to evaluate one of these rules as an example, okay? I want to focus on Rule R2.
What, what, what table is this? Is this 7-9?
Seat at the table.
7, 9th.
AI, the Modern Approach Book.
or simply print the PDF file that I just used, okay? It's one and the same thing.
So I'm looking now at Rule.
our group.
B11.
Double application, P12 or P21. That's the rule.
I'm going to evaluate the rule. 4.
the values. There is one row that has Sure, if I'll wait.
Or… values 312 is equal to false.
NP21.
Is equal to false.
Rich?
So, can you tell me something about this guy, if this is the case?
This evaluates to what?
Pawns or… pawns?
is launched.
Okay, this means that P12 or P21, is false.
Okay.
In this row.
So I picked up a row, right, a corresponding kind of row. I didn't write down now, however, which row did they pick.
I think I picked up…
B12, P21… Anyway, in that, I will tell you, I probably need to find out offline. In that row that I picked, B11 has a specific value, okay? It can be either true or false. Here, I have noted down that B11 is false.
Therefore, We have false double implication force.
Right?
So… Force double implication force is… force implies force, and force implies force, which is… True.
Therefore, That's it.
R2.
Evaluates to true.
So, I'm evaluating similarly all of the rules R1 divided for all permutations that I have in my table.
All of them. 128 of them. I'm evaluating these wires.
And… Now that I'm finished with the evaluation of the values.
I can fill in this column over here. This column will only be drew only when These guys are…
Where is that? Here, here. These guys are true.
So only a very small subset, right, of the models make the knowledge base rule, satisfy the rule.
For this subset that satisfies the knowledge base, what the knowledge base knows about my request. What is my request?
You're telling me how some books.
What is the request? They're requesting? Not P12. Not P12.
The P12, is false. Therefore, the not P12 is true. So, the knowledge base will respond with true.
are we following what's happening? Okay, so that's a,
I am not going to,
move on to the syntactical way, because I only have 5 minutes. So I'm going to stay here 5 minutes, because I know it will take 5 minutes for you to sign that you are here, right? And then close the discussion for today. Who hasn't signed on this one? Okay, please come forward and sign your presence over here.
Thank you?
There was any discrepancy of the original.
Right now.
However, if you want to.
But anyhow, hold on.
I'm not happy.