Joining in this kind of remote setting.
I will, hopefully,
recover soon and be, back at NYU next Friday.
I am,
what I'm gonna do today is I am going to continue after we typically review, as we do, the material we covered last week.
I will continue, and I will finish the transformer architectures.
And, then I will…
At the end, I am going to, also try to
Cover some material, not really a lot, out of,
Computer vision kind of textbook, just to give you an idea about how transformers are also applied in computer vision.
As we will see, they are, kind of very similar in terms of architectures.
But the interpretation there is kind of slightly different, so I'm just gonna focus on that.
I am also going to cover some material which is not necessarily inside your course site, which I'm trying to make sure that it is going to be.
During the weekend, or by next week, so at least you have a chance to review it.
It has to… it's a new material that has to do with a mixture of experts, and how the
How they're connected with the,
And sample methods that we have,
met, in, residual networks. And then, if time allows, I will enter, the next topic, which is,
I want to… I wanted to do a topic on neurosymbolic reasoning, but definitely I will start with symbolic reasoning, or logical reasoning.
And then, obviously, the neurosymbolic reasoning is definitely something that, we don't have material on, but it's something that's definitely of interest.
as of late, to reduce hallucinations, and I will provide, if you want, some references on that front. After that, we'll go to planning.
Planning without interactions, and planning with interactions, which is the reinforcement learning, and conclude this course.
That's, that's the plan for the next, few weeks.
Anyway, I just wanted to provide that. I know that you don't have an assignment to work on, definitely, however, you have your project, so I…
I know that it's kind of difficult to have a Q&A, especially if you cannot really… none of us can see each other, so I will, I will refrain to ask questions about your project and how…
much progress you have made until next week. I hope you have started it.
Nevertheless, and I hope you have, selected your partner, and I hope your… everything is… Progressing fine.
Okay, if you have, again, any questions, please don't hesitate to interrupt me.
You should know I cannot really see your chat messages, because I'm just having one screen over here.
So let's get started. Any questions before we start? Any concerns?
Arjun Bajpai: Professor, can you please also tell us about the midsection marks distribution?
Arjun Bajpai: Later.
So this is not, you know, it's definitely something that the TAs can,
can comment on that. I don't have the numbers in front of me. But the midterm was,
As I said, the midterm is, was probably, like a bloodbath of some sort.
So…
It was the distribution, if I remember right, was… the median was kind of low. I think this was on the 60 or 65, something like that.
And, the maximum grade in the class was, like, 87.
And only few people achieved in the neighborhood of 80. So, as I said before, if you are kind of,
wondering what on earth is going to happen with my grade. As I said, the…
letter grade is, going to… I mean, you… the bad situation you can find yourself in is if you did badly while the rest of the class did very well, and it seems that's not the case. It seems that many people did below expectations.
And the rest of the class also did below expectations. So, effectively, this means that the thresholds will move,
In a direction that, at the end.
Mind you, that the midterm is 20% of the total grade. The thresholds, nevertheless, because of that, will move into the right direction.
To make sure that we have the same distribution of letter grades as we were planning to do that anyway, okay? So, that's all I can offer at this point.
Any other questions with respect to grades or anything else?
Okay, all right, so, last week, I think we, started with a simple RNN kind of layer. I think, the discussion was kind of pretty straightforward. The week before that, we had some discussion about the fact that we needed to have some band of vector hidden states.
the vector hidden states effectively encompass the whole, sort of, range of Latin variables that are maybe needed for us to represent, the structure of a time series.
and time series as a, as represented by M examples, that is, not just one.
We then proceeded into writing down the equation of a simple RNN layer, and contrasted it with the equation of the dense layer.
And finally, we recognize the fact that we need to decouple the, obviously, the hidden state from the prediction, and the prediction in our case.
Back then, it was a scalar, therefore we had to use some kind of dimensionality reduction step.
Followed by a dot product to produce this kind of scalar. As I said, we could produce the scalar straight away.
But nevertheless, we created this diagram that, to also showcase the case of, classification.
And this place of classification is going to be needed every time that you are going to be using these kind of structures to do other tasks, such as language modeling.
And others.
So, with this kind of head, we kind of understood that
If you put everything kind of together in just two snapshots in terms of time.
of this kind of unrolled kind of unrolling architecture, that the backward propagation, that is now called backward propagation through time, has a critical kind of determinant in what is going to happen in the matrix W prime.
That we've seen in the architecture.
And, based on the eigen composition of this kind of matrix, then we can actually have
Gradients, which are, propagating,
in a way of a diminishing way, or exploding way. I went into your website, and I kind of showcased the simple scalar kind of equivalent. It's an abstraction anyway, nevertheless.
That showcases that as well in the… in the simplest possible block diagram you've seen in… that is also known outside of this kind of domain as infinite input response filters.
We then, went into…
Sort of trying to understand how we can potentially correct, increase the memory, or at least control
the flow of the gradient in this kind of earlier kind of simple architecture, and we came up with a long, short-term memory architecture that borrows ideas from highway networks via the line that connects
And the input, straight to the output, except from the fact that in this line, we also select what to propagate
Forward, in… Please, please go on mute if you're not on mute.
sorry, what to propagate forward via this kind of forget gate. So the forget gates and the other gates over there, they are equipped with segomodal outputs.
units. These sigmoidal output units are vector-in-vector output sigmoidal units. Therefore, every element of that kind of vector is, is a number between 0 and 1, and this means I have the
capability of… Propagating anything on the input.
With respect to the… what we call the short-term hidden state, and also, in the output, again, with respect to the short-term hidden state, but also, via the forget gate, the long.
memory that we call a shorter… a long-term memory that we call the S of T.
And, and we didn't really spend a lot of time on this architecture, as, you know, it also was kind of a labor architecture, a full understanding of this.
with examples and so on, would actually take significant amounts of time. LSTMs, for the purpose of natural language processing, are not used per se. However, as I mentioned, the architecture of recurrency
together with some kind of combination of state-space methods that we have seen in the hidden Markov models, are, led to the definition of a competitive two-transformer architecture called Mamba.
Nevertheless, we have not really spent any time on that, and we advised you to look at
a video from the course site that kind of explains a little bit with animations the function of the LSDM network using a simple, very, very simple example, in fact.
Then we transitioned into language modeling with this kind of RNN architectures. The language modeling was a prediction, obviously, task that had to do with probability distribution of predicting the next
Token, given a context of tokens.
And in a very trivial, if you like, block diagram, indeed, the previous N tokens, in this case, are coming in at the input, at the bottom of this kind of diagram. They're going through embedding kind of layers, and they are forming either long-term or short-term hidden states, or both.
and to propagate them forward that, that leads to a prediction, okay? The prediction of the token that comes after that context.
And so with this, kind of architecture.
In mind, we will see how this equivalent kind of language model is implemented with
the transformer architecture, but it was kind of instructive to see what was happening in just a plain language model. I think I advised you to look at a character-level language model with RNNs.
that it is in the website. I think I now remember that there was some kind of formatting issues on the website, which I will…
I will need to… Write down, to remember, to change… to revisit the website and make these changes.
for that specific page. Hold on, let me just write it down, because if I don't write it down, I'll just forget.
Today is the 7th.
Servants.
Okay, noted. The neural machine translation was another task that I wanted to spend some time on.
And the reason is that it is definitely of a representative, an earlier representative of an encoder and decoder architecture. So in neuromassine Translation, we have effectively a sequence-to-a-sequence model that we are going to be doing. I did spend a lot of time on it.
I provided kind of an example, with, 3 source tokens and 3 target tokens.
from a foreigner language and translated into an English language.
we recognize a couple of things. I think there was also a question about… that I think also… I don't have an answer right now, but I think I alluded to some answer with respect to what is the training protocol there. I will come back to you on that.
The encoder at its own here in orange is effectively producing a thought vector by looking bidirectionally the hidden states produced by the corresponding RNN layer there, RNN network there.
We also did not cover the fact that RNN network can also be deep, therefore it may have stacking of these RNNs one on top of each other. But nevertheless, here we just see a shallow RNN layer
That it is, unrolled.
Producing the thought vector phi, and it is really this single thought vector phi which is going to lead us
believe it or not, to generating all the translations which is needed for that specific context of the source language. So one vector generates
All other vectors that you see here in the translation, which is
Obviously, something that, can be improved, and it has been improved with corresponding transformer architecture, which are now encoder-decoder architectures in transformers.
So we, basically called, we recognized a couple of things that were actually going there. The… the fact that we have some form of teacher forcing.
Where we effectively feed, during training the ground truth, for each of the decoder kind of steps, and obviously, in inference, we cannot. Therefore, we are feeding and routing the previous translated token as
the token at the input of the next unrolled stage of the decoder RNN. And then finally, we clarified why we need bi-directionality, why this could be helpful for us.
In this, kind of, architectures.
But not only architectures that have to do with RNN, but also in transformer architectures. In many instances, we are looking at multiple decodings, and this multiplicity of decodings was explained here via the… the rationale of that was explained here via the concept of maximum likelihood sequence estimation.
or MLSE, also known as the ViterB Decoder in other domains.
Every smartphone that you're holding right now has a VW decoder, in fact, inside it.
The, and I gave, if you like, an example, which I… I hope you remember it, but effectively, I was showingcase how, by keeping track of multiple trajectories, of decoding trajectories, I'm able to, calculate the likelihood
Of a sequence, rather than, apply the…
memoryless, greedy approach in producing the output token out of a softmax, and although the proof was not actually given, MLSE is the thing to do, the optimal thing to do, rather than the…
greedy, single… single, stage kind of, decoding, as… if I may call it like this.
So, that was basically, the… and you will see frequently this being quoted as BIM search, and the hyperparameter of the number of branches also quoted as, as such, tool for you to select.
Okay, so this was basically the beam search kind of algorithm, which is helping us to produce the right kind of decodings across, let's say, sequences, and I also provided a blue metric here as, for your information. Obviously, there are tens and tens of metrics, and I…
An exhaustive enumeration of this kind of metrics is not of any value. However, I recognize the fact that we
Need to have at least some idea of, of something that is actually pretty…
popular. I'm not sure if it is 100% kind of useful nowadays, but it has been one of the kind of a key NLP metrics earlier. I provided an example here where the blue
Metric has taken the semantics of precision.
And of course, there are others that also capsule recall.
But I didn't mention any of that.
So that was the sort of, metric over their discussion. And then I transitioned to Transformers.
which effectively will replace the RNNs for the language modeling task we are primarily interested in. The transformers eliminated the recurrent connections, therefore introduce the need to
capture the…
position of every input token, because now the recurrency is actually gone, and therefore everything is fed to that transformer architecture in parallel, which is largely a permutation invariant kind of architecture. If you don't
Explicitly have positional abettings, something that we'll cover today.
We need to…
separate the transformer in our minds into, effectively, two things. One is the attention mechanism, which is
primarily the main innovation about the transformer architecture, followed by a head
Well, the attention mechanism is going to lead to some kind of stacked architecture of multiple attention mechanisms, building, representations, and then, finally, the head is attached to the transformer, on top of this kind of attention layers.
That will actually finally implement for us the prediction of the next token.
I motivated a discussion on transformer architecture by looking at three sentences.
The first sentence, and all of these sentences had exactly the same token. However, each token had carried to a quite different meaning. Animal tolerance and a sports team, and I started to explain how we can potentially
move the token, which right now sits on exactly the same position in this D-dimensional space, into 3 different positions. And each of these three different positions, the blue, the orange, and the green, will actually
be in such point, such that the meaning of the token is the one which is conveyed by our sort of interpretation of each of these kind of sentences. Animal tolerance and sports team.
So.
I think we are on… we are not on mute, so let me see if I can, if I can mute everyone.
Can I mute everyone from here?
Yes, I can.
Okay, so, so I started, having,
this kind of, you know, mechanization of what I just mentioned.
Initially covering the simplest form of the attention mechanism, which is, we are going to revisit today.
And, in that kind of attention mechanism, we had,
a matrix, capital X, that we stacked all our, context tokens.
our tokens are… have a dimensionality D, and the context size is capital T.
Obviously, I had to select some small numbers, that will allow us to
see what is really going on here. The small letter D, obviously, is into the several hundreds of dimensions in practice, and the capital T nowadays can be, tens of thousands of
of tokens in there, the contact sizes. Okay, so the… in this kind of, simpler kind of attention mechanism, we are effectively trying to use the similarity kind of metric to assist each token go into the corresponding location.
That we are interested in. So, the… another way to look at the similarity kind of metric, is that, which other token can actually
positioned me into that kind of appropriate location, so we did…
some kind of a weighted sum. This weighted sum is obviously implemented here by a dot product, X times X transpose.
Give us a score matrix, and this kind of score matrix, we wanted to, sort of normalize every single row of this kind of score matrix, that's that.
It is a number between 0 and 1, and in addition, all
attention weights sum to one. In other words, we are effectively introducing some form of competition between the tokens in the context that it will allow only a few, or many, depends on the situation, to actually help us via this kind of attention weights.
So that it is basically in the… in the orange box, you can actually see.
the, sort of weighted summertime just mentioned.
That mapped the token Xi to a location, to a new location, XI hat.
Right? And in terms of a matrix form, the X hat matrix can be written as actually shown also in the orange box, and obviously this is called a self-attention mechanism, because inside that kind of X hat matrix, also the… our own kind of token
Is there, and in fact, our own token will receive help from other tokens in the context, but also will provide help to the other
Tokens in the context. That's what's called a self-attention, a simple self-attention mechanism.
So that is, I think, where we stopped, from our kind of treatment of the transformer architecture. So if I may draw now a kind of a light line over here.
I wanted to ask you if there are any questions on everything I just mentioned today.
The day is the 7th.
Okay.
Okay.
Alright, so let's move on now. Let's now try to see…
Why this kind of attention kind of mechanism may not necessarily be,
May need to be enhanced, in fact.
So I want to know, actually, you to recognize the very fact that the position of the token.
So, let me write it down. Up to this moment in time.
the position… of XI hat.
is determined
Entirely.
by… the context-free.
embedding vectors.
Off.
the capital X matrix.
So I hope you recognize that, because even the attention weights are themselves a function of the X.
So… so the… everything we got actually, so… so effectively, this is a fairly deterministic.
way that you can actually, if you have an Xi, in a very deterministic way, and you have a context in a very deterministic way, this location of the Xi hat is, is fully determined, okay? And, so what we would like to actually to do
Is to also Notice a little bit, some other
potential ways that could potentially change that kind of location. So we would like to, first of all, change that, so the enhancement
is for… the transformer itself
Because these context-free abaddings were actually set from something which is external to the transformer. In fact, we saw a technique that provided this contextual-free abettings earlier in the previous kind of lecture.
So… What we would like to do is to allow the transformer itself, to, to determine.
The location.
of Xi hats.
In other words, to determine the contextuality of this kind of, context-free kind of vectors.
In a trainable way.
In that way.
That is trainable.
And in other words, there would be some kind of a parameterization in that determination of this location, so…
Let me write it down, write it down as well.
parameterization, Wolf.
the location.
of Xi hat.
So we… we need to introduce some kind of parameters here. Another way to kind of introduce this kind of… to understand a little bit the need for being a little bit more, to have a little bit more flexibility than the simple attention mechanism kind of affords to us, is to notice that, the first sentence
if I may kind of repeat to the kind of three sentences over here.
I will… the first sentence was, you know, I love bears.
The second sentence is C… Bears the pain.
And the third sentence is, bears.
Won the game.
And actually, you can notice that the meaning of the bears over here is,
I mean, if I go and kind of start labeling the…
sort of tokens over here. Definitely the eye is a subject.
The love is obviously a verb.
And the bears is obviously an object.
I hope you can see that, despite the fact I'm not able to
to sense if you're understanding everything I'm saying here, please interrupt me if I… something is not clear. So the…
Obviously, the… in the second kind of sentence, the bears
Is, takes the role of the verb.
And, obviously, the subject is here, and this is basically the object.
And in the third kind of sentence.
Bears takes the form of a subject.
Okay, so this is, one thing that,
we actually can notice there, even within this kind of pattern, even within the… I will call it the same kind of linguistic pattern of subject followed by verb, followed by kind of an object, the…
Function.
That the bears is actually, taken is,
It's kind of a different, even within this kind of pattern, and to some degree.
And this is actually a safe kind of assumption. The function is heavily correlated, so let me write it down.
in, languages in general, the function… Of, each token.
Ease.
Kylie.
correlated.
2… each position.
In… The… linguistic.
Not there, but a linguistic pattern, because we may have many.
So many, many, sort of, in this kind of linguistic pattern, you know, I hope this was actually clear. I don't have any… many other examples, because I'm not a linguist. What I'm trying to do here is to
Sort of, you know, motivate the discussion of…
what is the form of this parameterization? Okay, and I went ahead and I kind of labeled this kind of tokens with some kind of role that they are playing. I called it a function.
I could have called it a role.
As well, and, and, this kind of, labeling is…
Obviously, no one is doing it in practice, but it's actually… it helps us
to understand, a lot of things about the architecture of the transformer, the… of them.
More elaborate kind of,
sort of mechanism that I'm about to describe. So, given all these kind of requirements, the… first of all, the parameterization, therefore, the
The need to… Map.
The location of this,
of these tokens from an internal mechanism, a transformer kind of architecture, I can actually go and imagine a situation where somehow I need to convey
the role.
In this kind of parameterization. So, for example, I need to find a way that I can convey the role of object
In the first sentence, I want to find a way that this,
Verbness is, present in the… in the location of this, bearish token in the…
second sentence, and in the third sentence, I want to…
convey some form of subjectness on the… on… with respect to this kind of token. Fair enough up to this moment in time, so let me write it down. I want to…
I'm going slow because I want…
to go slow at this moment, and it was a critical point to understand several terms that I'm about to introduce. I want to be able
Via.
this parameterization.
To convey.
the role.
each token.
the role… That each token.
Will.
Lay.
June?
the attention mechanism.
Okay, so if I kind of introduce some kind of a… coordinate system.
So, this is, like, kind of a key here to understand. So, if I introduce a coordinate system.
where I'm going and sort of labeling, and this is basically a D-dimensional, so I'm,
I'm just drawing it like this in a kind of attempt to actually have multiple kind of axes. And I go and manually
Label the axis as the axis of objectness.
The other axis could be the axis of verbness.
And maybe this guy is adjectiveness.
And I have D of them, I have D of this access to label. Obviously, no one is labeling them. I'm labeling because I'm now extracting things, rather than… the machine is not going to labeling this… with these things anyway, but I just want you to label… to label it because I want to convey something subject-ness.
The labels that the machine is using is 1 to D. That's basically what the machine is actually gonna do.
So, if I am to imagine now that, that every token is, effectively
wants to convey its role, right? But also, not only convey its role, but also convey a meaning of,
I… what I'm looking for mostly to… help myself map
myself into the right place. So, you can understand that, out of this other con… as far as bears is concerned, out of the other tokens in this, first sentence, which one you think that it will help most bears
map itself into a good place. A good place meaning a place where it will be associated with
with…
the meaning of an animal. Which of the other two? We have only two tokens here, I and love.
Which of the two tokens you think will help mostly the token bears?
That's a question for you.
I will… Risk the question.
Despite the fact that I will be getting crickets, but I hope you… someone responds.
Philippa Scroggins: Whatever you prefer?
It will be the firm. That… so love…
Love is going to be, quite instrumental
In, in the, in the determining, if you like, that bears has now the meaning of animals. And, over here in the second kind of sentence, bears.
Most likely, out of all the other remaining tokens in the context, most likely, it will…
Wants to receive, help from the token pain.
most of the help that will be coming from Togan Payne. And, in the third sentence, probably either the verb or the object will actually be, over there, the one or the game will actually be, equally, sort of helping Bears, the sports team.
to extract its meaning. And this is actually a kind of a key observation, that it will allow us to now define this parameterization.
So, any questions before we proceed?
Okay, so, given all that we just discussed, I can imagine that a token
each token in my context, and I will basically
draw now the tokens is, in some halving, effectively, each one corresponds to this kind of box.
Okay, and each token kind of a meeting, and I'll tell you how it will emit them.
Three Psalmi formation.
First, the first piece is… Cold.
a vector, That we call a query.
and asks the same question I asked you A few seconds ago.
what I'm looking for to receive help from, mostly.
At the same time, I need to convey to the others
what am I? And this token will be called
This vector, sorry, will be called a key.
And, I want to decouple.
The mechanism of attention, I want to decouple.
the red arrow, From… a starting point.
that I will be calling a value. So, what I called earlier as Xi, I want to be able myself to determine that it's going to be the most appropriate Xi for me.
Okay, so I want to be able to have the flexibility for the attention mechanism to determine the second vector… the third vector over here that I'll be calling a value.
And it's gonna be the starting point.
Via which the attachment mechanism will take that value and map it into the right place.
Okay? So, inside this kind of value, it may already have some Notion of meaning.
but obviously not a complete notion of meaning. And this notion of… this meaning is something that is internally going to be determined by the transformer. So, I will explain now
The situation now, what is going to happen when all of the tokens are effectively conveying
With a different kind of index over here.
that same information, all of them are doing exactly the same.
Any questions up to this moment in time? This is… this is the time to ask questions. This is, I hope the discussion was with this kind of labels that I kind of invented here, right?
as I said, this labeling is not really happening in the machine, is kind of understand what I'm attempting to do. What I'm attempting to do is to suggest that a parameterization, and this parameterization is going to be implemented as follows.
I will… I will extract the query vectors via… a linear projection.
I will extract the key vectors via yet another linear projection.
And I will extract the value vectors with yet a third linear projection. And, if you are
If you are wondering how come a linear projection, or three linear projections will actually achieve what I have
of what I'm trying to achieve over here. I wanted to remind you
What we have seen, two lectures ago.
When we are actually looking at the… at least linear projection.
So remember that, back then we said, oh, let me take a vector that I call here the center world, and let me map it to a space, to a place in this kind of D-dimensional space, okay? This was a projection.
And the projection, the resulting vector was called Z back then. And this projection had very good properties.
And these good properties were extracted
Were produced, so not extracted, were produced
from, you know, a couple of mattresses here, the W and the W prime.
So, there is a precedent that someone can say, you know what, if the mattresses are the right ones, I can actually do a lot of things, a lot of useful things in that, in that neighborhood of…
where the Z vector is falling, right? So I can… I can take this kind of X vector at the input, which I call WT over there, and I can map it into a neighborhood which I want to
Sort of carry the meaning, let's say, of a car.
or in other words, right? So that… that same, thing here will actually happen, in the… in this kind of elaborate attention mechanism, but in a slightly kind of different way, because now.
I have to take this X and map it into… into a place where it will carry, two, definitely two meanings,
one meaning is going to be the query, and the other is going to be the key. Well, let's look at the first example here. In the first sentence.
Where is going to be the… the bears will emit that it is an object?
Okay? It will emit strong objectness, kind of, identity, if you like. The identity will be that of an object. Therefore.
Do you agree?
that in the very first sentence, bears, and I will call her now these bears.
With this color over here. Did I use the same colors? I want to make sure that I used… yes.
I will… I will color these bears to be orange over here.
Verb?
And, one color this guy to be blue.
Okay.
So… In the axis… in this kind of new coordinate system, because, you know, a projection
is introducing a new coordinate system. I hope everyone is familiar with it, and everyone is familiar with
whatever we discussed with embeddings a few… a couple of lectures ago, and also with concepts like principal component analysis, which are basically linear projections. So, if I… if I ask you to plot a vector that conveys
A lot of, What is this? Objectness?
With respect to the identity of the first
bears, sentence, in the first sentence. Would you agree that you will,
That you will plot something like this.
Or some vector in any way that has significant component, along the… Objectness axis.
Okay? Can you confirm that you understand what I just said?
So that this… this… Green kind of arrow.
Is effectively the key.
that, the… token bears will convey to the rest of the tokens.
Okay.
I am… I am an object, for I'm aligned along the objectness axis.
It maybe has components about others as well, but… because it's a statistical machine anyway, but…
I hope this is clear, okay? In the other keys that we see over here, in this kind of a new coordinate system.
Will be from, bears being a verb.
bears being a verb. So, most likely, in the second sentence, the key for BERS will be probably something like that.
And in the third sentence, what is this? Probably it is a subject, yes, so it will be, you know.
Something like that.
I hope the colors kind of help.
December gate, yes, go ahead, go ahead.
Ojas Gramopadhye: I have a question. So, when we say that the key for bears is gonna be, like, this particular thing, so what exactly is the key, like, I mean, what exactly does it mean?
I will tell you, I will tell you. I will tell you how to form these vectors now. So, I will, as I said, I will use linear operations to actually do the projection. So, I will do
The following.
I will do, I will form all of my query keys by taking the…
context-free embeddings, matrix X, and projecting them into this new coordinate system using a trainable matrix, WQ.
I will do the same for the keys.
a different matrix, WK, is going to be used there, and I will do exactly the same for what we call the values, with yet another third different matrix. So, effectively.
No matter where the X, the context-free.
A bedding vector of bears.
Was, right? From the… you know, before we looked at this kind of attention mechanism anyway.
It is effectively mapped To a key that is determined by this operation.
Okay? Obviously, the others' tokens are also mapped.
in the new coordinate system somewhere. So… as the…
training goes by, and this is really the key observation here. As the training goes by.
what is going to happen in this kind of a D-dimensional space, which I will kind of draw over here with this kind of abstract kind of cloud.
if…
that… that, the vector that corresponds to the query of bears. What is really the query vector of bears? The query vector of bears is…
I will call it, let's say.
UI, you know, for lack of a better kind of symbol.
will… needs to receive help from verbs, correct? I mean, that…
That position, it will be a long.
the objectness, right? So, sorry, it will be along, the verb-ness kind of axis, because the query bears
the bear's token in the first sentence, wants to receive help
mostly from verbs, so it is asking, hey, I'm looking for a verb.
Okay? And what is going to happen? The… keys…
that are going to help it, and it may not be only one, could be many. Record could be, let's say, K, J, K, some other indices, you know, vectors.
If it is to help it.
Do you think that the dot product between this query and the key will be small or large?
So what I'm asking is, very clearly the following question.
If I am to help you, bears, and I am a verb, I want to position myself
Near to you, or far away from you.
That is the near.
Mayor, which means that As this thing is trained, and sees many
of these kind of tokens, you know.
many instances of bears, and many instances where bears is a… is a… is a… is an object, and in this kind of instances of where there's an object, it will position the keys
that are… That are, that can help out.
That kind of verbs, that kind of token verb, to put… to, mostly, the keys that will show up next to that query.
So, in this kind of D-dimensional space, Everything changes.
However, this… change is an average change across potentially many, many, many trillions
Of tokens that this thing is going to see.
Okay, so the exact position of the keys, And the queries cannot be
I mean, I provided some kind of a simplistic view of the world kind of here, but of course, as you can understand, that's actually a synthetic example to say, hey, I am an object, that's my K over here, the green one over here, and I'm looking for a verb, and
I will, I can denote this thing to avoid this kind of confusion by suggesting I'm looking at the first sentence.
I'm looking at the first sentence, where I'm an object, as far as that's concerned.
And I'm looking for a quer… I'm looking for a verb, therefore I'm going to be issuing a query.
I'm gonna be issuing a query.
That it is going to be… Something like that. Q.
1.
Right? If I have…
If I have a verb, I mean, the query says, I want to see a verb. I'm looking for a verb. It's almost like a romantic advertisement, you know? I'm a male, 23, and I'm looking for a female of age, you know, 20, whatever.
So, I'm looking for a… for a verb, and I am, like,
I'm an object, okay? And all of the verbs that can help me
what they will do? They will position themselves next to me.
Next to my… query vector.
Are we following what's happening?
Katherine Zhuo: Oh, not really.
Okay, alright, so let me try again.
In the first sentence, I asked the question, the…
bears transmits the information that I am an object. And I asked the question, which of the other two tokens
It's going to mostly help me.
And you answered?
Love.
That's what you answered.
Okay? What is the… identity for…
What is the key for, love?
The key is going to be along which axis?
is gonna be along. The verb box is correct.
Because it's a verb.
Okay, so… what is assistance mean? Assistance means that there's a large dot product, right, between
The object and the verb.
Because really, in the first sentence, the verb is going to help bears
sort of, have the meaning of an animal, right? So… that…
The verb and the object that the…
Sort of, the query.
That I'm looking for the verb.
And, the token that says, hey, I'm a verb.
should be close to each other, that's all I'm saying.
Does it help in any way, this explanation?
So every single token is…
emitting these kind of three things. And if I'm saying over here, A, I am looking for a verb.
So, I am, let me use another kind of color.
I'm looking for a variable over here.
All the other tokens that are going to… Help me.
which… how they… how they are helping it by emitting the information, A, I'm a verb.
Are going to create
Strong.
they will go… they will respond to it, to me. In other words, they will attend to me, right? And that's the term.
And, you can imagine that,
Not every single key is going to help me in the same kind of way.
Some of these lines are very thin, therefore the weights are going to be pretty small, and some of these, they will be pretty determinant of
in the response, and therefore the lines will be kind of thicker, you can imagine that, almost like creating some kind of a communication network between me, that I issue this kind of query, and the keys that can
Best respond to that query.
Shuts that.
the… A red arrow.
is determined. This self-attention kind of red arrow is determined. This is basically the first part of this kind of mechanism, the determination of that kind of red arrow, right? And obviously.
That red arrow will be used to take me from
this specific kind of value, location, into what we call a V-hat.
location.
In exactly the same way. So, let me just also…
draw the other thing that I…
Drew, kind of, earlier. So the…
I'm starting with some kind of a V, let's say VI. I'm determining, with keys and queries and attention mechanism.
that it will position me now to another location called VI at
Okay, so this is basically, learnable.
attention mechanism. That is, instead of using the dot product of XX transpose, we'll actually… so I'm going from the XX transpose dot product.
To a dot product called Q.
K transpose.
Which is, obviously, Q is… I told you it is XWQ.
Where I'm issuing all these kind of queries, and I'm forming a dot product with all the keys that can actually help me.
Which, of course, means that this dot product will actually do this.
X.
WQ.
This exposition here… This transposition here is… W, K transpose.
X transpose.
So already we see a similar kind of thing with this, what we've seen kind of earlier, but over here we have this kind of two matrices, and this is the reason why this is called a generalized dot product.
So this generalized dot product.
This generalizer.product is the one that defines this kind of a new line now, because the queries and the keys
R.
Determined after a training kind of process.
in a way that I described, For every query.
A whole bunch of keys will show up to help.
And they will take the value, which is also trainable, of every token, and move it to the VI hut.
Is this better now? Any questions? Any…
This is the point… this is the point where we have to really…
understand what is going to happen. After seeing a lot of these kind of tokens, the W matrices will be determined in a very similar way as we did with the contextual-free embeddings, obviously.
Ojas Gramopadhye: Oh, Professor?
Yes, quote.
Ojas Gramopadhye: Can you once again, explain the role of, like, I mean, which roles are, which, roles are being taken by the keys, and which roles are being taken in by the queries, with this example again?
the key says, okay, I'm… I'm, as far as the token is concerned, I am an object, right?
The key for bears in the second sentence will say, okay, I'm also, you know, I'm having a verb as a key, right? And then the third sentence, I am a subject, right?
Again, this is the instructive version of the things that are happening, right? In the first sentence, I'm issuing a query that I'm looking for a verb.
In the second sentence, I'm issuing also a query that I'm looking for an object, and in the third sentence, I'm issuing a query that I'm looking for either the verb or an object, somewhere in between.
To help me out.
Ojas Gramopadhye: Right, okay.
Okay, so that's basically the K.
and the queue, right? And obviously, I… I also… I want to decouple the red arrow, right? The… the… let me put it this way. I want to decouple the…
Direction of the red arrow, and the…
length of that red arrow, okay, from where the VI starting point is, right? Which is get another third projection.
Now, in other words, the VI will… although the Q and the K says, A, I'm an object and I'm a verb, the V may say, you know what, I am not sure exactly what I am right now, but I'm increasingly believe that I am an animal.
If I may want to say it that way, okay? I'm… I'm seeing these bears being in a… somewhere in between an animal and, and, and,
a sort of, what is called a tolerance, and a sports team kind of thing, right? And so this is what the V is, right? And it will be mapped
into contextual locations by this kind of learnable attention mechanism. Yeah.
Ojas Gramopadhye: Makes sense, thank you.
Okay, okay.
I'm glad that even remotely, we managed to…
do something here. So, let me just draw the block diagram. The block diagram will take this kind of X,
And it will, it will,
you know, using the WQ matrix, the W, K matrix, and the W… Vmatrix.
as inputs, it will create…
the Q matrix, which has dimensions T by D, It will create…
The K matrix was dimensions T by D.
And it will create the V matrix, well, dimensions T by D.
And, we'll take these two, form a dot product, which is, QK transpose.
to form the score matrix, as we called it earlier, which is now T by T. This is a potentially huge matrix.
And, and then it will, do an operation which, I will clarify what it does.
It's like, A normalization kind of operation, divide, in other words.
the elements of this S matrix by 1 over square root of D.
And pass it through to form the attention of weights, it will pass it the result through the softmax.
to form the capital A matrix, which is obviously also T by T.
And the elements of the soft… of this capital A matrix will multiply.
the… V matrix.
to form.
The matter is called V hat.
which is T by D.
which is, evidently A, which is T by T by V.
Which is T by D to form the T by D matrix, as you can actually see here. And this is basically the… all the…
all the tokens now find themselves in the typical, usual dimensions of D by D, inside that kind of VHAT matrix.
Okay, so first of all, two things that are… need to be kind of clarified. You know, first one is the… what is the division?
the 1 over square root of D is actually doing here.
Okay, and there's an analytical way to actually show what it does, if you assume some kind of a… kind of a variance one, kind of inputs.
and information of the dot product, but I will show you what effectively is going on if it is not there, okay? And this is basically in the core site.
Because I have a picture over there.
that I want to go, first of all, to clarify this kind of point.
So, if I go back to the core site,
And, this is, what is it? Is it here?
I'm gonna go into… This, single-head self-attention.
And, I want to…
You to look at this kind of a scaling that, operation, which actually this thing is actually doing.
And, I have effectively two plots here. One, in the kind of, first plot.
is we have, let's say, instantiating, if you like, a random vector over here, because the softmax is seeing, effectively, a random vector at the input, and this kind of random vector is the original kind of random vector generated, as you can see from the source code, with NumPy random.
Gaussian kind of random vector at the input, with, let's say, 8 elements. And in the second kind of case, I took this kind of random vector, and I kind of multiplied it with 100.
Okay.
So…
Effectively, what we actually try to do, to convey is what happens after this softmax. What is the output of the softmax? So, in the case where we have, I will call it normalized kind of inputs, the softmax is, producing
definitely, a situation, because remember, the softmax is producing the attention weights, where lots of other tokens are actually helping us position ourselves, if you like, right? So the attention, the attention weights are kind of
determined from a wide range of other tokens in my kind of context. In the case where I don't have this kind of normalization, this is manifested at the extreme with this multiplication by 100, the softmax output is very, very sparse. Basically, effectively, what it does, it basically says, okay, only
Only a very few tokens will be able to help you.
And I want definitely to go into the blue.
situation over here, and have the kind of the softmax, have many of the other tokens kind of express some kind of opinion as to where I should be in this kind of D-dimensional space. So that is really the role of that kind of a scaling, that it kind of plays.
And, and that's basically, you know, something that I…
Sort of, wanted to show
There is some kind of analytical way of showing it, if you assume Gaussianity and kind of a standard deviation of one.
And the fact that I have a dot product, that it's going to increase the variance by D, and therefore the squared of D is bringing back the variance to 1, but I'm bypassing all that, and just show you what is happening using a numerical kind of example.
Any questions up to this moment in time?
This kind of, first step.
Okay, so what we are, wanted also to mention here is that, we definitely have,
I will call it another requirement here for decoder kind of based architectures, right? And this requirement comes from the fact that, you know, I cannot… I mean, while during training.
all my tokens are available to me, right? Both past and future. During inference, I will have only tokens that I already have seen that actually can assist me.
And therefore, I want to make sure that I don't really cheat.
And and therefore, what we're actually going to do is we need to implement some form of masking.
And, this means that I want to… when I'm actually, I'm, let's say, in position
8.
I am going to receive help only from positions, let's say, 1 to 7, and I'm limiting the kind of context to 8 at this moment in time. But when I'm positioned, let's say, 7, I can only receive attention from 1 to 6,
when I'm positioned 5 from 1 to 4, and so on and so on. So what I'm… what… one way of actually doing that is to introduce, sort of, obviously, to make this attention wage for these specific tokens zero.
And, or alternatively, I can actually go and make this, the input to the softmax a very large negative number, and therefore, I will… the output of the softmax for these things will actually be zero. So therefore, I'm… this kind of,
A trick allows me to, which is called masking, allows me to achieve the aim to avoid
future tokens, which I have not seen yet, to…
helped me out. Okay, so that's… that's basically the… One kind of,
sort of a stage that I want to… so that's why this attention mechanisms are called masked Self-attention mechanism.
Okay, so, so that is, that is basically what's happening in this, in the, in this master self-attention mechanism. And then finally.
this is basically the end of this kind of what I call the… so up to this moment in time, so the second thing that I actually also covered is called masking.
Where?
you know, keys…
that correspond.
Two future tokens.
Cardi… Attention.
weights… of 0.
Okay, that's the thing, and so for details, see the core site.
And so everything… so all of this kind of block diagram is called, this block diagram.
Come on, this block diagram is called, a single head.
Shorty. Hi.
like, kind of…
So this is called the… Single… head… Self-attention.
And what I, would like now to take you a little bit through is, the need for,
multiple heads, kind of in parallel.
So the… the thing I wanted to explain is the so-called multi-head.
Self-attention.
And the need for multi-head self-attention can be intuitively, again, being, sort of,
Even if when we are looking at convolutional neural networks, remember there's kind of a role of the filters, and I was telling you that we need, if you like, multiple filters to be active.
In order for us to extract different spatial kind of patterns that are present in my input image. For similar kind of reasons, I want you to think kind of this way.
We can actually have, multiple single-head self-attention mechanisms.
So that's basically the heads.
The block diagram survey, which we actually have
Plotted, the balloon of which we had just, just finished.
So these were gonna be, effectively working in parallel. This is basically my, capital X matrices.
That,
Each one, of course, will be, given… I'm just gonna only write one. It's going to be given by the specific,
values of Q, K, and V.
Indexed by H. H is the head index.
And the outputs of these guys are going to be… concatenated.
And, lead.
I'm gonna draw it like this.
via a… I will call it a mixing matrix.
Which is, D by D.
to produce at T by D.
matrix.
That is, going to be my V-hat again.
And this is, this kind of, the way that actually you can actually understand that is that…
I… right now, I kind of looked at one kind of grammatical kind of pattern, or some kind of a temporal correlation kind of pattern, but I want to be able to
flexibly look for others as well. Obviously, we don't have just one pattern, we have many grammatical kind of patterns, and each single head could be dedicated to
Sort of, extracting some kind of,
sequential correlation or temporal correlation, you can see it in the case of, language.
that corresponds to this kind of, that requires different, potentially different matrices of Q, K, and V, to be used.
So the… the… one thing that we need to be very careful
Is, first of all, let me write down the heads.
Are working in parallel.
To extract.
Let me call them temporal…
Well, I don't want to call it temporal correlation, to bring it parallel to extract.
Dude… extract.
representations.
the tokens.
Across.
different.
I want to be careful when I say grammatical patterns. Again, I'm using this kind of objectness and verb-ness and things like that, adjectiveness and things like that.
To use, you know, just from instructive kind of point of view. And, over here, there's not anyone kind of, again, labeling this pattern as grammatical pattern 1, 2, and 3.
and, and, concatenate, extract the representation of the tokens across different grammatical patterns. Finally, and at the end.
create… Composite.
Representation.
using… a linear… Combination.
Of the single head.
Self-attention, or masked self-attention.
kids.
With, which are represented here with a capital H. So, in the notations, a capital H is the number of heads.
This is the number of heads.
I can have multiple heads.
not, hundreds, but I can have, tens, potentially. Definitely, in architectures, we started with, 8?
If I remember right. In fact, I have
have somewhere here some kind of a table, which, of course, now I'm not able to find.
But, maybe I can't find it.
No, it's impossible to find it over here.
But I… you know, there's a… for every architecture that is being used in the transformers, for lines, language models, there is a… there is a number there for the capital H.
Okay, so this is basically the… what is called the multi-heart self-attention.
And, I'm going to enclose it now with yet another kind of block. We are not really done yet.
Unfortunately.
So this is the… multi-head self-attention, MHSA, okay? And this is, now going to…
Up to this moment in time, probably you.
notice that every single operation we have been covering up to this moment in time is linear, right? So, now the time comes to introduce
nonlinearities in the building up of this kind of representations. But before we actually go into this kind of nonlinearities, there is one block that it is,
kind of, very typical to find it inside this platform architecture. It's not really a…
The most essential block, but for training purposes, it helps a lot.
And this is called, layer normal… layer normalization.
So, what you will see in, in, very typical to see, layered normalization.
Nowadays, this is also called RMS normalization. It's kind of similar in terms of function.
Where effectively this X goes through.
the layer normalization, or explain what the layer normalization is actually doing. And it's led to whatever the number of heads that we have.
the… well… I should have plotted,
This branching is happening inside the multi-head self-attension, so let me remove it.
So this is the multi-head.
self-attention that I just, drew.
And, here we see also, well known to us, Skip connection.
To ensure that,
to ensure, if you like, for the same reasons as we have seen, kind of, earlier in the press nets.
To ensure, kind of, gradient flow.
to build.
If you like, let me call this,
Z, for a lack of a better kind of letter.
And, so this is, this is basically the…
block diagram that kind of surrounds the multi-head self-attention mechanism there. Decoration of multi-head self-attention is happening with layer-normalized inputs and the skip connection.
And, one…
If you're wondering what is this layer normalization is doing, and how it is different from what we have seen already.
Now, actually, I'm forgetting if we have seen batch normalization in this class. Did we cover batch normalization at all?
Do you remember anyone?
You don't.
Okay? If we… if you don't remember, most likely not.
I… Anyway, it doesn't really matter. I will review it.
I will review what batch normalization was doing. So, I would just want to open a parenthesis here to
Sort of cover, batch normalization.
The doc… the discussion became… Kind of a block diagram oriented, so not…
You know, it's not really necessarily very deep, but it has to be done.
So what is really batch normalization? So, remember, one, kind of a key
Observation we have seen when we looked at backward propagation.
At some point, I was, I was drawing.
this kind of sigmoidal, unit, I was calling this white hat.
And I was doing this W, and I was doing this kind of X, and this is a dot product like this, right? And I was asking you.
To tell me, what is the gradient.
of Y hat.
With respect, 2W.
I do hope that you remember this exercise.
Yes?
Come on.
It happened several… several weeks ago.
At the beginning of this kind of, exciting course.
Where is it?
backward obligation, remember I was thinking with… no, this is after.
Background obligation, sigma hearing, fully connected.
back propagation… Do you… do you see this exercise, Lecture 4?
9, 26, 25.
It should be in your notes. What was the key result there?
The key result is that the gradient
of the output of a neuron with respect to W is proportional to To what?
To the inputs, to the inputs, right? So…
Okay, is this, kind of a rocket science? No, it's not a groundbreaking result, but I just want to write it, here.
the gradient…
is, this gradient, the gradient, this gradient is proportional
to the input.
which is X.
Now, I want you to make this kind of extrapolation, okay? I know it's not scientific extrapolation by any stretch of imagination, but now let's take this to a layer now, when we have now multiple of this kind of neurons.
And you can actually make the same kind of claim, and if I have a neural network with some form of,
You know, again, it's a very gross, kind of, you know.
diagram over here. We have multiple layers, one after another.
And then finally I have a head.
Right? To produce, if you like, a white hat.
And I will have a head in this transformer architecture as well for the language model kind of prediction.
Every single, the input leaves, produces a hidden state, H1. H1 comes at the input, produces another hidden state, H2, and so on and so on. 2 is plenty for making, if you like, the argument.
Since the gradient flow, and this is where the extrapolation is, is dependent on the…
Input to a layer.
Do I have the control of that input?
And the answer is… Do I have control of the input?
And the answer is no, because that input depends on the trainable parameters of that
layer L1. So what L2 is seeing is… definitely depends on what L1 is doing.
Right? As far as the L2 is concerned, I don't have any control of what L1 will give me. I hope you agree with that, right? So what is really batch normalization is actually doing, it says, you know what, I want to give you some control.
And, obviously the control is not going for you to determine, but I will find out the best
setting.
And what is the setting? Imagine now this X to be a Gaussian random variable with some kind of mean and some kind of variance, right? So I want to, if I'm… if I plot
The unified kind of,
sort of, situation over here. I want… Whatever comes here, To have a specific mean.
And a specific variance.
And the same thing across the whole network.
Okay.
To have a specific,
Are you following what's happening? I want to have that capability.
And I want to… this mu and sigma 2 squared, to be…
trainable parameters. So these guys are trainable.
And the reason why it has to be trainable is, at the end of the day.
the network can only know what are the right values here as… here are. And I can do this batch normalization.
By kind of estimating what is coming in,
Normalizing it to standard deviation of 1 and mean of 0, let's say.
And then have, these trainable parameters, which are scaling and, shifting.
this, standardized, if you like, Xs or H's.
to the right place, which I want to…
to position them. Okay, so the batch normalization as a kind of a formula looks a little bit strange.
Where on earth is this now? Looks a little bit strange.
In the sense that it produces an XCAT with an estimate of mean and an estimate of variance, which is, you know, across the batch.
of, that is coming in, right, for every, kind of, training batch that's actually coming in. So it produces an X hat, which, pushes it to, be a standardized variable, and, then, it does the scaling.
Which is the gamma coefficient here, in this second equation, and the shifting, which is the beta
coefficient here in this kind of, in the same equation, to the right place. The beta and the gammas are trainable.
And, so the Gaussian, if we want to think about, like, a single wired gun of Gaussian, moves to the right position, such that the task that this neural network is actually going to be doing is fine. The gradient flow, in other words, will be fine. If this layer, once this
input to be in that kind of place in the… for the… for this Layer 2 to receive this H1 in the right place. That's a job of the bunch normalization that is actually happening in the L1 layer.
And the batch normalization can happen after the activation function, or before the activation function. If it happens before the ReLU, effectively what it does, it effectively controls the clipping probability, because at the end of the day, the ReLU will allow
only positive values to go through, and this means that it will clip all the negative kind of values, so controlling this clipping probability will
helped. If it is done after, It effectively shifts their clipped
Gaussian, let's call it like this, the clipped Gaussian, to be on the right place. I hope everything is kind of, not, you know.
completely foreigner to this kind of concept of how I move a Gaussian around, right? So I'm moving it with some shifting and some kind of scaling. That shifting is scalable… scaling is trainable.
Are we… are we following here? What's going on?
Any, any questions?
So that is the batch normalization. And what is really the layer normalization comes and says, you know, I want to do the same thing.
But you know what? I'm not too sure that I can pack many examples in the batch here.
Or, I'm not too sure how many I will be able to pack in a batch, because over here, I'm dealing with a situation where the contact sizes are huge.
the… and therefore, I have hardware constraints in terms of my VRAM that I am able to accommodate.
I may have a situation where I need to serve multiple requests that are coming staggered in time, as far as during, let's say, inference. And, so the batch sizes are kind of pretty limited here. Okay, so I will… I don't… I cannot really extract sufficient statistics from,
within the batch to be kind of accurate, so what I'm going to do is I'm just going to do exactly the same thing as batch normalization is doing, but almost like in a transposed way. I'm going to do the estimates based on my
Dimensions of the features.
So what you see here in the figure as kind of C-channels, that's a feature dimension, and so I'm not really doing these, estimates based on the batch dimension, but
I'm doing it over the feature dimension.
Okay, so that is really the difference between patch normalization and layer normalization, and in the extreme, I'm able to accommodate just one token.
Just one, one example that comes at the input to, to, to be, served.
by the… a transformer.
So, layer normalization, you know, achieving
the same… Thing as… Batch normalization.
By, averaging.
But by extracting.
the mean.
And standard deviation.
Across.
feature dimensions.
Your normalization is actually going to be applied across the rows of this kind of,
capital X kind of matrix.
And, and then it will, it will feed it… fill the result into the…
Multi-head self-attention.
And, what the multi-held self-attention output then will actually be, this, this, Z that I called it earlier.
This is the… this is… so basically, this is… let's… let me call this capital A.
block diagram, and let me call this, capital B.
block diagram. This Z will actually come in, it will
Now we… now we will introduce the nonlinear component.
For… in a similar kind of…
way of thinking, as we have been discussing a bit, kind of, earlier, up to this moment in time, we have no only linear operations involved.
in the attention kind of mechanism, and now this is a time to form nonlinear transformations, so this is what an MLP is going to do.
multi-layer pessertron, so another layer of… normalizer.
It's going to feed.
the input of the MLP. Obviously, you need to have the normalizers for
the multiple states involved in this attention mechanism to form, finally, with the skip connection.
What we called the… Let me call this the… V-Hot.
Okay.
Let me call this the V-Hut.
So, the concatenation of A and B, you can actually see them in parallel. So, in series, A plus B, the concatenation of those two blocks is what is called the masked
a transformer layer.
And this master formula layer is, going to be, not only one, but just like what we have done kind of earlier, is, we can actually have a multiplicity of this. So we have… let me call this MTL, for lack of a better term. So this is MTL1.
That will feed to another MTL2.
Let's say, and so on. And then finally, we are kind of done with this kind of,
representation, buildings.
of, the tokens.
And this is basically what we called earlier the body.
And and, and then…
we need to attach a head to do the task, and the head, just like what we have seen earlier, it's a linear combiner of the features delivered by the body. So the linear combiner is
is, the…
and another kind of linear matrix, let me call this kind of W, followed by a soft marks. I don't have space for the softmax, so I'm going to do it over here.
And this soft marks is across… My vocabulary kind of size.
to, to… to provide to me the…
Y hat, which is now the post-zero probability
across all tokens, we have vocabulary. Okay, so this is basically,
The posterior probabilities allow me to select
in a gritty way, or according to the maximum likelihood, sequence estimation kind of criterion, the,
the next token.
what is the best next token to… to produce, okay? So, many of… I mean, these layers are, you know, could be many. I mean, 32 is a typical kind of number. I can't remember now exactly every single
Language model has its own kind of number over there, but 32 kind of layers are kind of a typical number that comes to mind.
Okay, this is basically… This is the… This is the whole story.
In a kind of a…
It's kind of a dry discussion, but, you know, this is a block diagram of the transformer.
Architecture, and with one notable exception, we have not discussed yet about, positional embeddings.
All this discussion up to this moment in time came, was done with this capital X matrix.
that,
That, obviously, if you change the rows of this capital MX matrix, you will arrive in exactly the same outcome.
So this is the definition of permutation invariance.
What we will do, what we need to do is to close or complete this kind of discussion, is to talk about
the positional embeddings.
That carry information about which position
The order via which the auto cans kind of show up.
So, in your courtside, you will actually see
There are multiple methods for positional counterfeit beddings. One of them is a method that it is
a little bit more difficult to understand, so I'm going to focus on that.
Allowing you to study the second method in your course site, which is basically the learnable embeddings kind of method.
This first method I will call it the…
The first method is called the Fourier method.
Because it has a lot to do with,
sinusoids and stuff like that. So let me just bring up some notes here.
But avoid dying from memory.
Okay, so… In terms of the… of this kind of method.
we are going to be forming a new X that I will call, for lack of a better term.
Tilde.
So I'm gonna form…
for the i-th kind of token, which is going to also be, obviously, at the input of the transformer after this positional embedding, I'm gonna… I'm gonna add to that context-free embedding vector. I started everything today, a positional embedding vector, which I will call
R… Aye.
Okay, so this is basically the original.
Context-free vector.
And this is the positional.
encoding vector, which I have to come up with.
Okay, so, I mean, obviously, someone could actually ask the question.
you know, why addition, okay? And, what is the alternative? Well, the alternative is to say, okay, I'll just concatenate
the positional encoding vector to the convex-free vector. Obviously.
The second situation is possible, okay? The only, obviously, a thing that can happen is, that I'm gonna be increasing
the D dimension.
And, as you probably noticed, that kind of a D-dimension is also,
part of the equation of the complexity of the transformation, and so people suggested that I can get away with an addition if
I can achieve two things. One, I do not want to disrupt
considerably, the context-free kind of vector, so if my…
addition is not disruptive, then that's kind of good news. And the second thing, as I said, is that since I'm going to be doing a lot of linear operations in this kind of attention kind of mechanism.
I want to do it… The linear kind of operations around doing the multiplications, and so on, are…
Somehow, they need to… They're preserving,
the addition operation. So, if I multiply with a matrix.
the summation of the original kind of matrix plus the positional and quantity kind of vectors, then, I… there is obviously a way to separate the two, in this kind of associated kind of rule, and so things are okay. And another thing I wanted to
Sort of make sure also is that these positional kind of encoding vectors are
yeah, as I said, they are not going to be…
Sort of disrupting, moving, in other words, these contextual free, vectors.
Changing them completely, from what they were… they were.
So… I…
Okay, so… so what we can actually, need to focus a little bit our attention now is,
you know, everyone who's looking at this kind of Fourier method, they will find in front of them a specific kind of formula, which is initially, almost
difficult, or even impossible to understand why on earth the formula is the one it
which is quoted, and which is… which is the following. So my… Nth element.
Of the positional encoding vector, so this is basically the index of the token.
And the n is the nth element.
of the… of the vector R.
is, equal to…
There is this sinusoidal function. The details are not really important, but the graph, what these results to, is kind of important.
And, this is, this is basically, that is, Let me see here…
in the formula.
So there is,
There's a number L, okay, so there's L,
divided by D.
For them to be even.
And there's a cosinus.
of I.
L… N minus 1.
divided by D.
N is equal to odd.
So…
Let's see now.
So, L…
I will confirm this a bit later, but I don't think that my notation here is the right one, because in my notes, I'm just realizing this, this L
This is T.
Capital T.
I will confirm this. This is basically the, the contact size.
So, I will confirm this.
Let me write it all, however, this way to make sure that all the variables have been kind of defined over here, but more importantly, if you plot this, if you plot these kind of functions, because there are functions that is obviously a function of the…
position.
And and so I'm going to plot a kind of a diagram, where in the bottom over here, there will be
the elements of the R vector, okay? So this is basically, the first element,
Okay, so this is, R1.
to… R2, and so on and so on.
Okay? And, and the y-axis here is… so…
So this is basically D over here, D elements I have here. And, I have over here the…
Because, you know, the length of the positional and conicado vector has to be, you know, the…
by 1. This is basically the same dimensions as the other vector, the Xi over here. So that's why this, the columns is D, and the rows is, effectively positions.
Those are positions. Okay. So, if you plot this,
If you blow this, you will actually see
I will dare the blood.
I'm sorry, I don't have the plot very accurate here, but there is some kind of…
point that I want to make.
So… So, the plot is, is going to result into this guy over here to be,
the highest possible kind of frequency. This guy to be one octave kind of lower frequency, this guy to be one octave even lower than the previous frequency, and so on and so on.
So, you know, people have said, okay, fine, if I… if you are in this kind of position, for example, this is the position, position, let's say, 3, you draw a line.
This line cuts this kind of functions at specific kind of points.
And you go ahead and read the values of these functions for this specific, position that you are. So the position is actually changing, and the
the eye is… is, going to… Be picked up by…
It's going to… it's going to refer to that.
position, right? So this is basically the… Capital T over here.
and yet another position. If you are in another position, you grow another line. You have different values of this vector kind of R, D values, in fact, and then you will add them to, you know.
you will add them to the X. And, you know, no one kind of understands exactly
what is going on here. So… After some,
After some kind of thought, and they're…
And the explanation came, from,
a very good explanation came from Bishop's book, called Deep Learning.
which showed up kind of recently, is as follows. Since you have positions over here, this is basically positions.
And you have D dimensions over there. Go ahead, and I'm going to just do D to be a small number. I don't know how many it will end up. Let's say this is enough.
Go ahead and say, okay, what is…
if you want to designate with binary digits the position, what would you do? Well, this position is definitely 000…
1.
Then, as you can start filling in.
Oh.
Right?
As you're filling in these kind of numbers, what do you notice?
The first column here.
Is this the highest possible frequency in this kind of digital domain that you can think of, that you can see? Yes, it is 10101010. That's the highest possible frequency.
So let's call this frequency F.
this guy… As you probably see, is one octave is F over 2.
This guy is… F over 4. Do you see that?
If I could… I could have continued.
You will get that, 0, 0.
0111, 0, and so on, right? So…
if you see this kind of pattern, it will actually, go along this kind of line. So, effectively, this pattern that you see here is the analog
Version of, what you do… You would have intuitively
Did if someone asks you to encode the position with binary digits.
Okay, so this is the… Kind of, encoding.
of position.
with… D?
Bye, Nutty.
digits.
And, this is encoding.
of position.
In… the analog.
Domain?
that if thresholded.
Will, definitely.
Thresholded meaning digitized.
Result?
in this.
The advantage of this kind of analog kind of representation is that, because of the cosigners and assigners are, you know, numbers between
less than one, right? Then, effectively.
You don't necessarily hit this kind of a concaxory vector with
Always a 0 or a 1, but you are… you're kind of picking up numbers which are in the
Between minus 1 and 1.
That's the explanation of this four-year kind of method.
for positional… positional encodings. So I hope the equivalence between the two kind of helps you kind of understand, what is actually happening there.
And any questions?
Okay, no questions?
You understand everything? Okay.
Okay,
All right, so what else, we need to say here? I want to spend, sometime…
Trying to… Sort of give you some kind of idea about,
Kind of new… newer kind of transformer architectures,
Well, there are two categories of discussions we can have. Obviously, we can talk about many lectures about transformers and
Even more lecture about large language models, but that's not necessarily a course kind of dedicated to
sort of NLP, so I'm trying to hit some kind of main points. One of the major kind of innovations the last few years, especially after the introduction of DeepSeq and, kind of a new reinforcement learning approach, is
This came into the, sort of, foreground very strongly, is this kind of a mixture of experts.
That had some kind of significant impact, not necessarily fundamental impact on the kind of architecture, per se, but, you know, certainly had some significant kind of impact on the efficiency, of, running these things, obviously in this
You know, very…
very difficult situation where we are, where we have some really high expensive, very, very expensive kind of memory in these accelerators that we try to
Preserve if… if… If not, because we have some kind of almost
duopoly kind of situation in this kind of experience, at least at this moment in time. So this kind of mixture of experts has been, has found that it offers quite significant benefits, and I wanted to discuss the mixture of experts, not from the block diagram kind of perspective.
Because I'm not really good in drawing block diagrams, but just to contrast it with what we have seen earlier about unsample methods.
Because there are some kind of strongly correlated kind of things that, that,
That we have to recognize here.
Okay, so what is really the mixture of experts?
So remember these ensemble methods where we found them? We found them where? Where did we see ensemble methods in front of us?
Yes.
Ojas Gramopadhye: Residential?
ResNets, right? So this came out of this kind of ResNet discussion.
And, back then, I think, we did some kind of,
superficial kind of treatment of this kind of topic, where effectively I was telling you that the sample methods is, methods where they introduce, some kind of weak predictors. These weak predictors are
Sort of, going to…
In aggregate, they will actually offer, prediction that it is,
Much, much better than the individual ones, and
back then, also, I was telling you that what is the worst case, and what is the…
best-case situation you can have with this kind of committee methods or sample methods, right? So, similar, similar comments can be done for this kind of mixture of experts, but this mixture of experts are slightly different than the sample methods. So, I want to highlight the difference, so at least you're aware of what's really happening. So.
So I wanted to…
I want to make sure that I define this mixture of experts, first of all.
Is, as a conditional.
The infrastructure conditional and sample.
And what is really this, you know, a sample where we have… different experts.
That's why it's called mixture of experts.
This is the equivalent of weak predictors, right? Responsible
for… Different… parts.
of… P data.
Over the B data distribution.
So, as you can imagine this kind of P data distribution to be, obviously, a very, very complicated kind of probability distribution via which, of course, obviously, we are engaging a very, very complicated P model in order to capture it, right, to approximate it.
You can actually think about this complicated kind of P model as a model that consists of individual kind of P models, that we call here, kind of experts, each one tuned
to address a subset of this predator heart distribution. So…
Behind these kind of, scenes, there is a… A core kind of, concept called mixture of Gaussians.
So, think about.
Or emojis. Emojis are…
term that, acronym that's usually associated with this, mixture of Gaussian. So what is a mixture of Gaussian? What is a mixture of Gaussian probability distribution? Well, very simply.
is, remember I was telling you about Gaussian, that I wanted to come up with a P model from the samples?
You know, obviously we may have a mixture of these kind of Gaussians, where the probability distribution is going to correspond to some kind of a mixture of
you know, of these two, right? So, this probability distribution
could be a very, very complicated one. I mean, but obviously we see here a very, sort of, simple, univariate kind of abstraction about that, right? And in this kind of mixture of Gaussians, I have,
Gaussian components that are mixing together, that's why it's called a mixture of Gaussians. So we have two components, for example, over here, that each of… is with its own kind of a mean and parameters, right?
Invariance, but the 2, the mixing coefficient, is,
Appropriately chosen, such that a sample over here, let's say.
Has, is, the generation of this kind of sample to
components are responsible for it, so therefore, it has a responsibility kind of coefficient that is mixing together the two components. So, for example, if the two components has mean mu1 and sigma 1 squared, and the other component is mu2 and some other sigma 2 squared.
the… this kind of sample, we can suggest that it is produced from summing.
the two components with some kind of mixing coefficient, so it could be… Very simplistically, by, 1.
Sort of, normal of,
X, given nu1, comma sigma 1 squared, plus Glass.
pi 2 normal X given U2 sigma 2 squared, okay? So this is going to be my…
sort of, be data hot.
Okay? That is going to be the mixture of Gaussian kind of probable distribution. So, in the mixture of experts, a very similar thing is actually happening.
So, if, X is, so what's happening now in the mixture of experts? We have,
Let's say, some output.
Where… The output, the Y hat, in other words, is a prediction, is some kind of summation
let's say from i is equal to 1 to capital K, which is the number of experts.
a function called GI, which is associated with an input X, times a function f
I, which is… let's… let's say that's also a function of X. And what is this, function,
Fi of X is the expert.
In fact, it is the prediction of the experts, or the prediction.
of expert I, And, GI is… The,
Is a function of a ga- function.
of aggating.
Network.
In fact, let me call this to be the…
Yeah, it's a function of the gated network. Therefore, produces a number, right? It could be, associated with a softmax
Weight, similar to the attention weight.
Effectively, what it does, it says, okay, fine, I have a prediction, which is basically the aggregation, of,
of, ecoficien, yeah.
times the prediction of the expert, okay? The difference now between the two
Is the… the… what we had kind of earlier, and actually now, is that this,
function… let me actually, sorry, complete the definition here. So I have
The… from i is equal to 1, 2K.
GI of X is equal to 1, and obviously G
I of X is greater than or equal to 0.
So the… the difference here is that,
The, the, the gating kind of network,
It is a trainable kind of network, so…
It learns, effectively, how to weigh the experts, so the… the… the GI of X.
functions.
Function is trainable.
And the other difference is that while the weak predictors in the unsample method were
We're, sort of always trying to mimic the P data distribution with all… with those weak P-models.
In the experts, they don't care about this, they only care about the specific partition of the PData heart distribution.
Right? So that's why I kind of, sort of provided this kind of,
relevance to this kind of mixture of Gaussian distributions, because in mixture of Gaussian distributions, you have still, the pi 1 and the pi 2 kind of mixing coefficients that needs to be learned.
And obviously, the two components are kind of…
or more or less in this kind of univaroid kind of space, as you can see, have some kind of some soft responsibilities. I'm kind of only going to be concerned with everything here on the left, and everything here, the other component could take care of everything, everything there of the right. If I estimate,
my PI1 and Pi 2 kind of successfully, I have a very good P model overall, okay? That is…
That is, certainly what's happening in this kind of mixture of experts.
Evidently, you can, connect this to… What,
I was telling you a little bit about the best case and the worst case situation of having an ensemble, and equivalently create the best case and the worst case, because, you know, that situation that we looked at the ensemble kind of learning, I was telling you something about
that what is the best-case situation? Best case situation, every single member of my ensemble makes uncorrelated mistakes. That was the…
So… so the… the best case…
In both cases, right, in both a mixture of experts and ensemble methods, the best case is that every
Single.
member.
of the assemble.
Mix?
uncorrelated.
Mistakes.
2… The ones, the mistakes, in other words, Of the others.
So this is the best kind of case situation. In the worst case situation, I was telling you.
Worst case.
Is that, you know, everyone makes.
the same mistake.
And actually, you can see this kind of…
evidently, in the kind of, formulas that are associated with this. So what… what the… if you see this, kind of, in terms of formulas, in terms of the people who went ahead and quantified this, they, they used
regression.
and mean square error as the criterion. So, earlier.
the guys, like, you know, Ian Goodfellow.
and others in their book. If you open it, you will see
This kind of corresponding section where they go ahead and quantify the mean square error, that can result
If, we have, perfectly correlation, perfectly correlated mistakes, or uncorrelated mistakes, and therefore they define the upper bound
upper bound.
In terms of performance, and lower bound.
upper and low bound in terms of performance being upper and low bound in terms of mean squared error. What is actually, quite, quite kind of interesting is to repeat this kind of same exercise, when the criterion
in.
Mixture of experts.
When the criterion, when the criterion is… Crossentropy.
Okay, because it's really the cross-entropy criterion that we're actually using when we're actually doing language modeling.
Okay, and not only, I mean, you can think about language modeling, but also multimodal reasoning, all these other additional kind of models that they're using with visual language models, they're entirely based on exactly the same kind of principles, the same kind of architecture, but obviously a slightly different parameterization.
But this is the thing I wanted to cover.
For you.
Unfortunately, I'm kind of reaching kind of my limits here, and so if I can take a rain check, I can actually…
write it, up and, send this out to you so you can actually understand
What is really the mixture of expertise?
limits are, okay? And I think you would be, whenever you're studying an architecture like DeepSeek and others, that really involves, nowadays a lot of mixture of experts, a lot of architectures are going down this kind of direction, to at least have some kind of intuition as to what things you need to be designing for, okay, when it comes to
Criterias that not many people have addressed, earlier, like cross-entropy.
Okay, so this was basically, so, you know,
Let me, let me see, C right up.
That I'm going to send out to you, and then we can discuss it briefly the next time we meet, kind of next Friday.
So, that is basically it, for today. I don't want to expand into or open up a new kind of chapter.
I want… I… what I owe you also… is,
For next time is, you know, what happens with computer vision?
So, vision transformers.
or VITs.
You know, what is the difference there?
not much.
The interpretation obviously cannot really go through the same way that we did it over here, with verbs and subjects and objects, but the interpretation there is also very natural there, which I will provide.
And, also the interpretation of multiple heads is also very natural.
And I will cover that kind of next time, and close the topic of transformers, and then move into, symbolic reasoning.
Okay.
Any questions, any concerns, any comments?
Okay?
Well, thank you for attending.
And, I will, see you, next week, then.
Philippa Scroggins: Thank you.
cube.