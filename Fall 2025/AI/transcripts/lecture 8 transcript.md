Pantelis Monogioudis: Okay, it seems, today, is Halloween.
Pantelis Monogioudis: And, therefore, the attendance is, not very strong. But nevertheless, we will take attendance, even if it is a special day.
Pantelis Monogioudis: I will stop the lecture at 1.20, today.
Pantelis Monogioudis: And I'm not gonna have any breaks.
Pantelis Monogioudis: Because of that. Okay, so let's, try to see what, the logistical things that you, I think you are very aware of.
Pantelis Monogioudis: Your assignment is due on Sunday, right?
Pantelis Monogioudis: So… That's good. All right, so that's Assignment 3, and Assignment 4…
Pantelis Monogioudis: It's still missing, I guess. It's going to be, missing in action. It's going to be… probably be drafted sometime in the next,
Pantelis Monogioudis: couple of weeks. Your project was presented. I hope now everyone has a partner, or they select to do the project individually, right?
Pantelis Monogioudis: Either, either of the two is fine.
Pantelis Monogioudis: And, however, you should start thinking about your Milestone 1, especially after the deliverable of this Assignment 3.
Pantelis Monogioudis: So I'll position the assignment for it to be, well, it definitely has to be before Thanksgiving. Like, it's, like, like, mid… being November. I mean, we already have November, effectively? Like, is it tomorrow? Tomorrow, November? Okay, so… so I will…
Pantelis Monogioudis: send it out, two weeks, and then we are going to have the project. There will be no other assignment under Assignment 4, luckily, and so therefore, you will have plenty of time to deliver your project and prepare for the final exam.
Pantelis Monogioudis: Okay, alright, so let's see, last week, by the way, I heard a couple of people before we started that, they had a problem tracking the ball.
Pantelis Monogioudis: Okay, the bolo, obviously, I warned you that it is, an object that moves, a little bit,
Pantelis Monogioudis: in a strange way compared to players, right? It's much more difficult to track.
Pantelis Monogioudis: And, obviously, it has… you have to assume something different about this motion model compared to a player kind of motion model, especially when it comes to the motion… the transition variants, right? The transition model kind of covariance, where effectively.
Pantelis Monogioudis: If you spread it out a little bit, right, you're effectively assuming that the ball is an object that actually moves without really any constant speed, necessarily, right, or smoothly, because someone picks it, or someone is, the ball goes out of the
Pantelis Monogioudis: What is called the viewpoint of the camera.
Pantelis Monogioudis: and comes back. Now, the… comes back is one problem, but, tracking it while someone kicks it and goes into a completely different kind of direction, we cannot really do miracles here.
Pantelis Monogioudis: I mean, definitely we cannot do miracles. Of course, we will not penalize, you know, people who
Pantelis Monogioudis: we know that they tried, or it's evident on the positive strike. The goal has to be there for the ball as well, obviously, and they kind of failed to deliver a perfect striking situation. The ball is also featureless.
Pantelis Monogioudis: In a sense that, you know, it is…
Pantelis Monogioudis: obviously an object that you can be confused with any other ball, which is white, and therefore the re-identification piece is not that as strong of a case there, right? So, I think if you have the IDs kind of constant for the players, I think it would be much better.
Pantelis Monogioudis: Up to a point, of course, even though, because, as we discussed, in order to fully
Pantelis Monogioudis: tracking. You need to have multiple cameras, and you need to be able to
Pantelis Monogioudis: See the layers as they move in and out of each camera, which means that if we move out of one camera, they will go and be in the visual
Pantelis Monogioudis: a field of view of another camera, right? So, even those are a bit problematic, because they're sometimes shot from far away, and…
Pantelis Monogioudis: And, someone could go and,
Pantelis Monogioudis: because they have unique numbers on their back, someone could actually go and sort of read those numbers, or attempt to read those numbers, and then achieve something a little bit later than that. Then you also have a name.
Pantelis Monogioudis: On that thing. Anyway, many people work on video analytics for sports, so hopefully something came out of that assignment.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: Last time. Last time, we spent some time on tokenization.
Pantelis Monogioudis: What is… what does a tokenizer do?
Pantelis Monogioudis: And it calls words that end? It takes, it takes, input, some kind of, words, right? And, breaks it down into what?
Pantelis Monogioudis: Integrate numbers, ultimately, right? Because it will break it down into sub-words with this byte pair encoding. Now, during this discussion, the discussion is, of course, very superficial, so I want to mention a couple of things that,
Pantelis Monogioudis: That came to my mind after reviewing, reviewing the method.
Pantelis Monogioudis: So there was a question about, why we, selected age.
Pantelis Monogioudis: Next to, after S, right? And, there are plenty of tokenization kind of methods. I mentioned something about entropy, measurement, and, alluding to some kind of decision tree. Well, the method actually, in fact, it is, kind of similar to that.
Pantelis Monogioudis: the H is what happens to be the next letter after the S, right? So you…
Pantelis Monogioudis: select that as a letter. There's another… so behind the tokenization, after this course is done, and you have the winter break, if you don't have anything to do during the winter break, but if you, Shannon's.
Pantelis Monogioudis: source coding theorem.
Pantelis Monogioudis: Okay, the source coding theory is governing quite a lot of stuff that we are, we have in the AI in the form of generative AI.
Pantelis Monogioudis: It governs how we compress video, how we compress audio, how we represent, kind of, waveforms, with an underlying theory, which is called array distortion theory.
Pantelis Monogioudis: There isn't a red distortion theory behind this,
Pantelis Monogioudis: Because this is an encoder, a source encoder, effectively. You can read this later, and you can connect it into that purely very easily.
Pantelis Monogioudis: it so happens that this selection of this bipart encoding is a special case of something, deeper that is based on entropy, okay? And there's another technique called,
Pantelis Monogioudis: Minimum description length.
Pantelis Monogioudis: That is entirely based on, that entropic concept. And so, it just so happens that they selected this one for activity type of models, because it's easier and, probably, sort of, inefficient, okay?
Pantelis Monogioudis: However, when we go into the organization of video.
Pantelis Monogioudis: when you look at organization of things like video, then things are not going to be as easy, right? And therefore, you need some background.
Pantelis Monogioudis: To, on… from, from information theory to understand what is actually going on there with the codebooks that they have… they're designing to represent chunks of images over time.
Pantelis Monogioudis: Okay, so we're leaving certain stuff on the table, but anyway, the approach was kind of trivial. When do you stop this iterative way of merging stuff? Well, not interactive, but this step-by-step way of merging stuff, so when do you stop?
Pantelis Monogioudis: When you are… when you have a vocabulary size that you want. So you want a 10,000 vocabulary size.
Pantelis Monogioudis: you will, stop when, the recovery becomes 10,000. If you want 100,000, you stop when it becomes 100,000. Nothing, really, sort of, smart there in terms of, is this the most compact representation you can build in terms of
Pantelis Monogioudis: tokens, as I said, for the reasons I just explained.
Pantelis Monogioudis: Okay, so the… next we have this kind of a bedding discussion.
Pantelis Monogioudis: This ability in discussion was, Alright.
Pantelis Monogioudis: it was an evident in the kind of next step. We were seeing how our tokens from the vocabulary that the finizer kind of determined, and, we kind of,
Pantelis Monogioudis: Our job there is to map a token into a vector space.
Pantelis Monogioudis: That vector space, we call it R to the power of D, it's D-dimensional, and we organize why we need it. And we built… we converted the, embedding
Pantelis Monogioudis: Into a prediction problem.
Pantelis Monogioudis: So,
Pantelis Monogioudis: Based on the sort of idea that the meaning of a work is, primarily dependent from what shows up next to it.
Pantelis Monogioudis: We defined a window around every word, and at every step during, if you like, this kind of training kind of process, we have an input word, a center word, and we are trying here to predict the tokens that are next to it.
Pantelis Monogioudis: So we have a… we are… we are having a… the training data for this specific size is trivial to build.
Pantelis Monogioudis: Because, we have the text, and obviously we can see what words are next to it.
Pantelis Monogioudis: whatever world we are considering, whatever instance in time, and therefore we have a P-data distribution that, that tries to predict a joint probability distribution, a conditional condition on the center world.
Pantelis Monogioudis: So we, obviously, each one of these guys could be a very large dimension, could be 10,000, 100,000 dimensions, and things like that, right? So this conditional probability distribution is fairly complicated, and we try similar… following similar steps to what we have done in maximum life.
Pantelis Monogioudis: to simplify it, and write it effectively, with an IM-based assumption, which is
Pantelis Monogioudis: A conditional independence assumption, to write it as a product.
Pantelis Monogioudis: Of whatever pre-model.
Pantelis Monogioudis: we are going to construct. So we are following, maximal press principles, and, we looked at the laws that we have to use, and all that remains to be done is to come up with the P model.
Pantelis Monogioudis: Okay. The B model is going to be a neural network of some kind of architecture. That was the way we are building B models today. So we are… we came up to some construction such as this.
Pantelis Monogioudis: where… We are doing two things in this network architecture. We have bed and we lift.
Pantelis Monogioudis: Right? So we embed… The central world.
Pantelis Monogioudis: into a lower dimensional vector Z, which is my desired embedding at the end of the day. At the end of the day, being at the end of the training processes will be it.
Pantelis Monogioudis: And, subject to the fact that, if we lift it.
Pantelis Monogioudis: From a matrix, using a matrix with appropriate dimensions.
Pantelis Monogioudis: We will get a posterior provenance distribution that initially will be very noisy, but, gradually, after a while, for that specific center world, using this kind of mattresses, then, most of the time, the…
Pantelis Monogioudis: words of, that truly appeared in my P data distribution.
Pantelis Monogioudis: will appear next week, and therefore, we'll win the competition, and by minimizing this kind of loss, we will have an ability that
Pantelis Monogioudis: not… that is actually doing… not satisfying only this condition, but it satisfies some kind of aggregation of this… of this cross entropies, right? So, because this is now… now joined.
Pantelis Monogioudis: Right? We have a P model that has to be satisfying, sort of… it's not… the P model is not only doing just one prediction, it's doing multiple predictions at the same time, right? So we are doing joint optimization.
Pantelis Monogioudis: of all… The trainable parameters during the training, not one.
Pantelis Monogioudis: That's… that was the only thing.
Pantelis Monogioudis: Which was missing that discussion.
Pantelis Monogioudis: We disregard the other lifting mattresses, we only select them.
Pantelis Monogioudis: the… the betting matrix, the W star at the end of the day. And, we recognize that, because we have,
Pantelis Monogioudis: word that it is one code and call it at the input of the general network. All we'll do is to pick up the corresponding flow.
Pantelis Monogioudis: of that W-star matrix matrix, we will send this W-star matrix to a hub, together with some qualification of our training data set that we use to come up with this WSTARM.
Pantelis Monogioudis: So sometimes you see, given a betting matrix from, let's say, Hacking Trace Hub, or whatever that is, and you say, okay, this was done using a Wikipedia image.
Pantelis Monogioudis: Or whatever. So, so this was… this is a qualifier, we have to make sure, because obviously, if you did something different than Wikipedia English, obviously, you're going to have, even for the same token, a different application.
Pantelis Monogioudis: Because you're changing the P data, right? If you change the PData hat, you will have… you will be, in the training process, changing your P model.
Pantelis Monogioudis: I hope everything is kind of more or less understood in this kind of discussion.
Pantelis Monogioudis: We, recognize that we are going to have probably one and only one
Pantelis Monogioudis: building for the token bank, used as an example, that is going to carry inside it the influence of multiple winnings.
Pantelis Monogioudis: It would… it will be, during the training process, a mixture of, National Geographic, and
Pantelis Monogioudis: financial data.
Pantelis Monogioudis: Hello.
Pantelis Monogioudis: Any questions on this, surveyed infrastructure?
Pantelis Monogioudis: So…
Pantelis Monogioudis: We are then, jumped into, the next kind of, sort of, task, which was language modeling.
Pantelis Monogioudis: And up to this moment in time, we don't have a task, we just did the building, right? The representation, if you like. Now we are going to build, obviously, representations that will allow us to do predictions. Predictions of what? Of this nature.
Pantelis Monogioudis: Given the WT, if I give you a context before each.
Pantelis Monogioudis: So, we said, okay, fine, can we build a kind of a trivial kind of language model using, some kind of approaches that we call the recurring networks? Now, many people will, rush to,
Pantelis Monogioudis: to, push aside these recurrent neural networks, but I can tell you that they're not so easy to put aside, for two reasons. One is that they are going to be used even outside of the scope of
Pantelis Monogioudis: natural language processing, for, in general, time series predictions. And they're also, together with the state space models that we have done in the, like, hidden Markov models that we have done in the Kaldman filter, these guys, together with these abstractions, they are coming into,
Pantelis Monogioudis: Some competitive architectures to transformers, okay, today, and they are still being researched.
Pantelis Monogioudis: So, I think it's worthwhile, going, through that kind of trajectory and present Transformers as
Pantelis Monogioudis: That's,
Pantelis Monogioudis: as changes we are making into these, recurrent neural networks. Okay, so we started with, obviously we knew about what the state is. We started, by suggesting that we are going to be modeling
Pantelis Monogioudis: F.
Pantelis Monogioudis: The target function.
Pantelis Monogioudis: F, remember the mapping diagram, that it was, obviously, is a function of, of input, of X, we call it like this, back then. But in general, you, it's very customary to see this kind of dynamical systems, even if the function is not changing over time.
Pantelis Monogioudis: with this kind of letter of action, this notion of action. So action for us, for this discussion, will be the arrival of a token.
Pantelis Monogioudis: That's why X of t is going to be there. When you wrote down the modeling, the equivalent modeling equation. So we were going to have to come up with the G,
Pantelis Monogioudis: Then, with the appropriate theta, that will, allow us to, at this moment in time.
Pantelis Monogioudis: Give us, and, and…
Pantelis Monogioudis: a history, and why I'm saying history? Because if H of T, each H of T depends on HT-1, HD-1 depends on…
Pantelis Monogioudis: HD-2, and so on and so on. How long this history
Pantelis Monogioudis: And, we will call it, this thing, memory.
Pantelis Monogioudis: how long this memory will be, survived, it will, we will understand today.
Pantelis Monogioudis: And it could be connected to granular flow, okay, at the institution.
Pantelis Monogioudis: So memory is like gradient flow. But, not yet. All right, so we said, okay, fine, we already knew how to do this, right? So I started from the simplest possible trivial sigmoidal neuron.
Pantelis Monogioudis: that I already know how to do predictions without a dependency on the previous input, and I introduced a feedback stage.
Pantelis Monogioudis: And the feedback states kind of in what he said.
Pantelis Monogioudis: That memory has nothing to do with the memory of the gradient flow, which is just high. It's just, basically, it's a D3 flow. I store the complete… it's actually the actual physical memory. I store the HT because I'm going to use it as HC-1… sorry, I store the HC-1 because I'm going to use it when I produce H.
Pantelis Monogioudis: I also changed the sigmoider unit.
Pantelis Monogioudis: And the reason why I changed the single unit to a 10, it's kind of unique.
Pantelis Monogioudis: apart from being kind of historical, the reason, the tonnage units were also fairly popular back in the Rosenblatt kind of construction, this modern kind of Europe. But,
Pantelis Monogioudis: There's a model, units have, if you remember, have kind of a trust function that will look like this. So, you don't want to have, lots of zeros in the feedback loop.
Pantelis Monogioudis: So, I prefer to have positive and negative numbers.
Pantelis Monogioudis: So I cannot arrive in exactly equivalent if you ex… if you forget this thing.
Pantelis Monogioudis: Signal, predictor.
Pantelis Monogioudis: That, that I wrote exclusively like this.
Pantelis Monogioudis: Okay, that's fair enough. So this was basically with only the recurrent neuron.
Pantelis Monogioudis: And, we try to model a time series, problem with it.
Pantelis Monogioudis: So someone is actually giving you some time series of an asset.
Pantelis Monogioudis: And you were asking you to predict what is the price of the acid in the next interval, that, that obviously we have some kind of ground truth, but obviously, we require a predictor to…
Pantelis Monogioudis: will be as large as possible to that ground truth. It is a regression problem under the hood, but now considering the history of prices, right,
Pantelis Monogioudis: That, came before it. Evidently, we can do exactly the same, this prediction, by using your midterm
Pantelis Monogioudis: problem, question set, whatever that was, three, that to give you some kind of time series, and you were using a DNN, right, to do the prediction.
Pantelis Monogioudis: But over here, what we will do, since we are after,
Pantelis Monogioudis: A scalar, which is a prize.
Pantelis Monogioudis: Oh, it's, I cannot do anything about it. Surprise!
Pantelis Monogioudis: I will, reason about what's happening inside that single neuron by unrolling it over multiple instances
Pantelis Monogioudis: In time, back in time.
Pantelis Monogioudis: Okay, so what I mean by that is that physically.
Pantelis Monogioudis: At any one instance, if I not have this call.
Pantelis Monogioudis: then, basically, I have this… the same column, but nothing's changed at the input and output. So, I don't have…
Pantelis Monogioudis: I don't have physically a neural network such as this, but effectively, right, in this unrolled architecture, I do want to consider the fact that because of this dependency, right, that expands multiple kind of time steps, I can effectively have… effectively, what I have is a very long
Pantelis Monogioudis: We are a network.
Pantelis Monogioudis: That ultimately will produce a prediction, Not, however, straightaways.
Pantelis Monogioudis: But only after endious outputs are produced. That's the… that's the reason of this enrollment.
Pantelis Monogioudis: And, so at any one instance in time, I'm feeding the next token, or the next value, whatever that is, and I'll… I'll produce a Y hat.
Pantelis Monogioudis: Which is, obviously, I will train according to the mean square error, the percentage for Gaussian models. That is what I'm going to do.
Pantelis Monogioudis: In the regional kind of setting. And we have organized that, when I start…
Pantelis Monogioudis: Maybe the single neuron, he's,
Pantelis Monogioudis: not able to model accurately, because it actually lost the competition. If you look at the notebook, it lost the competition against
Pantelis Monogioudis: the…
Pantelis Monogioudis: sort of, dense network, so what's the point of going into studying this thing? We decided that, in a similar way as we increase the number of neurons to come up with a fully connected kind of layer.
Pantelis Monogioudis: We want to do two things. One, to increase the parameters, right, by putting multiple of these RNA neurons together.
Pantelis Monogioudis: And we had some discussion as to what should be the dimensionality, right, of this kind of number of neurons, because at the end of the day, this number of neurons
Pantelis Monogioudis: are going to form a hidden state now, the age that is not of a scalar, but it's actually a vector.
Pantelis Monogioudis: So that number of dimensions of that vector, we could potentially attribute it in a kind of an empirical way to, running an SVD on the data matrix.
Pantelis Monogioudis: Looking at a spectrum of singular values.
Pantelis Monogioudis: And, understanding.
Pantelis Monogioudis: That, we can actually get away with
Pantelis Monogioudis: Some dimensions, and keep only these dimensions.
Pantelis Monogioudis: So if you have a time series of, this number of features are in your… you want to consider a historic event.
Pantelis Monogioudis: Maybe not all data are going to be needed to be kept, right? And the reason why, you can understand it also from your kind of experience, maybe the price now depends on what happened last week, and maybe 2 weeks ago, and things like that, but maybe has much less dependency on what happened 3 months ago.
Pantelis Monogioudis: Okay, so there is some kind of a negative correlation.
Pantelis Monogioudis: In many times these kind of problems. And so we take advantage of that to suggest some kind of intuitive way of what should be this dimensionality, or whatever we'll call it, for that fission vector. But also, at the same time, we have to also solve the problem of, now I have a vector that I'm predicting.
Pantelis Monogioudis: But I, I want to predict not a vector, but a rice.
Pantelis Monogioudis: So I need to decouple, the… I need to take that kind of vector and do something with it to produce a Y hat, which is a scalar number.
Pantelis Monogioudis: Okay, so these are the two steps that we will do.
Pantelis Monogioudis: Alright, so… Let's start.
Pantelis Monogioudis: So this is,
Pantelis Monogioudis: What is the lecture today?
Pantelis Monogioudis: This is 10th, 13.
Pantelis Monogioudis: True date.
Pantelis Monogioudis: Okay, so I am going to start growing.
Pantelis Monogioudis: The layer now, which will involve multiple of these neurons.
Pantelis Monogioudis: And I'll write the equation of this layer.
Pantelis Monogioudis: Now, as you can see, we have now matrices involved, and very similar thinking as we had in a fully connected layer. We had the matrix involved there. Obviously, this thing produces a vector at the output.
Pantelis Monogioudis: And the equation that we would like to, model, I will write the equation, and then the… as I'm writing, you will see what the W matrix corresponds to.
Pantelis Monogioudis: this H of T, is equal to 10H, Oof.
Pantelis Monogioudis: You?
Pantelis Monogioudis: Which is a matrix.
Pantelis Monogioudis: of dimensions N, New Orleans?
Pantelis Monogioudis: by… and input.
Pantelis Monogioudis: times XD.
Pantelis Monogioudis: In other words, a matrix by vector multiplication, right? Which is obviously, XT is n input by 1.
Pantelis Monogioudis: by one.
Pantelis Monogioudis: a round of space, I will continue in the second line.
Pantelis Monogioudis: Blast.
Pantelis Monogioudis: W prime matrix.
Pantelis Monogioudis: Which is N neurons.
Pantelis Monogioudis: by a neurons.
Pantelis Monogioudis: HD-1.
Pantelis Monogioudis: Last B.
Pantelis Monogioudis: And, this B.
Pantelis Monogioudis: And this HD-1 is obviously 10 neurons by one.
Pantelis Monogioudis: N neurons is the dimension of my H vector.
Pantelis Monogioudis: And, where is this W matrix, coming in? The W matrix is defined as You transpose.
Pantelis Monogioudis: W prime transport.
Pantelis Monogioudis: Which, if you want to think about it, is going to be…
Pantelis Monogioudis: and input.
Pantelis Monogioudis: times N.
Pantelis Monogioudis: neurons.
Pantelis Monogioudis: This matrix is… is this guy over here.
Pantelis Monogioudis: N input times N neurons. And this W prime matrix, I told you, it is N neurons.
Pantelis Monogioudis: Yes.
Pantelis Monogioudis: E, with you.
Pantelis Monogioudis: N neurons times n neurons.
Pantelis Monogioudis: Okay, and so this one will give me, after concatenation, after concatenation will give me…
Pantelis Monogioudis: And input plus N.
Pantelis Monogioudis: Neurons?
Pantelis Monogioudis: Bye.
Pantelis Monogioudis: N?
Pantelis Monogioudis: neurons.
Pantelis Monogioudis: And the cocatonation over here, obviously, is… has…
Pantelis Monogioudis: N input by large N-neurons kind of dimension, so it is compatible with this W. Mocker, please.
Pantelis Monogioudis: To the concatenation of, so this equation, this equation can be written as
Pantelis Monogioudis: H of T is equal to tanh.
Pantelis Monogioudis: XT, H, T minus 1.
Pantelis Monogioudis: Dr. Yoon, Rasp B.
Pantelis Monogioudis: That is the simple.
Pantelis Monogioudis: Alright, man. Later.
Pantelis Monogioudis: And you can contrast it if you want.
Pantelis Monogioudis: with,
Pantelis Monogioudis: With the equation of…
Pantelis Monogioudis: the dense layer.
Pantelis Monogioudis: But he's always doing this one.
Pantelis Monogioudis: Obviously, it represents the dive, but the equation in purple represents exactly the dive.
Pantelis Monogioudis: Yes, go ahead. Scrolling up, yes, scrolling up. In the equation highlighted in yellow, what's this? So, so basically the orange enclosed equation is the main equation that I started a discussion, right? Then I wanted to define what is the W matrix that you see in the diagram, and I started…
Pantelis Monogioudis: To say, okay, what is my…
Pantelis Monogioudis: uMatrix, okay? The U matrix is…
Pantelis Monogioudis: N neurons by N input, right? So, therefore, the U-transpose is N input by N neurons.
Pantelis Monogioudis: And the W prime matrix was N neurons by N neurons, that remains the transposition to be N neurons by N neurons, right? And here, then, we are concatenating the rows.
Pantelis Monogioudis: of the U transpose and W prime transpose matrix. Because of the packet donation, we are adding the dimensions of the corresponding rows of these two mattresses.
Pantelis Monogioudis: And obviously, leaving the number of columns unchanged.
Pantelis Monogioudis: So this is the region of the last equation here in… brackets.
Pantelis Monogioudis: The input is not XT here, it is XT combined with HT-1, so the input is N input plus an nurse.
Pantelis Monogioudis: And what will happen with the…
Pantelis Monogioudis: the input of the 10X, in other words, will be a vector.
Pantelis Monogioudis: Right? Of, that results.
Pantelis Monogioudis: from the… W.
Pantelis Monogioudis: Times?
Pantelis Monogioudis: Sorry, it will result from this equation here.
Pantelis Monogioudis: The equation is incredible, right?
Pantelis Monogioudis: So, every nonlinear function will be applied element-wise.
Pantelis Monogioudis: If it is a vector at the input, so that NH is applied element-wise.
Pantelis Monogioudis: I could actually write it also as W times a combination of XT and HT-1, obviously, but then I'll remove the transpositions there, and I will do other changes, but in my notes, I have it like this, so it's a copy.
Pantelis Monogioudis: It's very difficult for me to think and write at the same time, especially if you're in presentation mode. But, any reasonable, mapping,
Pantelis Monogioudis: that, of considering the previous hidden vector now, it's, is, as an equation with the 108s, is acceptable for the ESP.
Pantelis Monogioudis: No formula for HT. Shouldn't be WXTHT minus 1.
Pantelis Monogioudis: Like, looking at it amongst the…
Pantelis Monogioudis: What is going on? So, XT, HT-1, has a dimension which is… N input plus N neurons.
Pantelis Monogioudis: We'll put XTHG minus one.
Pantelis Monogioudis: XDHD minus 1.
Pantelis Monogioudis: It will be what it will be. This guy is, okay, there may be a transposition kind of missing here, right? Or some swapping of W and this other vector, you want to say? Yeah, it may… it may happen, okay? That's why I warned you a little bit. Okay, so let's look at the… let's see if this equation is still valid. Okay, so XT will be…
Pantelis Monogioudis: Let me write it down. XT will be an input by 1.
Pantelis Monogioudis: And HD-1 will be A neurons by 1.
Pantelis Monogioudis: So, the concatenation, I want it to be…
Pantelis Monogioudis: N input plus a neurons, right? By 1.
Pantelis Monogioudis: And then I can swap the W matrix, and I can put it in front.
Pantelis Monogioudis: No, I don't want to put it in front. I want to, I want to… I can put in front of W transpose.com.
Pantelis Monogioudis: Okay, so I can… I can change the equation a little bit, right? To make it W transpose the concatenation of this
Pantelis Monogioudis: Two vectors, right?
Pantelis Monogioudis: Is that what your point is?
Pantelis Monogioudis: Yeah, there may be some dimensional tissue, but,
Pantelis Monogioudis: So I can, I can equivalently, okay, I can equivalently kind of write it as 10H, of,
Pantelis Monogioudis: W transpose, and W transpose is…
Pantelis Monogioudis: N neurons times N input plus N neurons, right?
Pantelis Monogioudis: And then you have the coatination of X, D, H, T minus 1.
Pantelis Monogioudis: Right?
Pantelis Monogioudis: Which has to be an input plus a neuros by 1.
Pantelis Monogioudis: Right?
Pantelis Monogioudis: So, one pop of salt.
Pantelis Monogioudis: Huh?
Pantelis Monogioudis: It's gonna be XT, HD in this one.
Pantelis Monogioudis: XT, HT-1 will be coordinated, not horizontally. Yeah, vertically. Vertically. Yeah, vertically, that's what… okay. Vertically, yeah. Let me, let me write it down.
Pantelis Monogioudis: Vertically, yes.
Pantelis Monogioudis: XT, H, T-1, right? Like this.
Pantelis Monogioudis: Last B.
Pantelis Monogioudis: That is a… that is, probably a better equation, okay? But I hope you got the idea here. It's not really to play so much with the dimensions as to get the idea that this is,
Pantelis Monogioudis: Obviously, you need to be correct with dimensions to make sure that our equations are correct, but I hope you got the idea of what is now the number of parameters of this RNN layer. The number of parameters is the W matrix and the B.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: That's the number of trainable parameters. We increase the number of trainable parameters, but effectively, we remain with the problem of mapping this H of P, which is going to be produced, after now unrolling the RNN layer after, let's say, a number of unrollings.
Pantelis Monogioudis: The time for making a prediction arises, let's say, at the 49th time, so I now have a vector of H of T.
Pantelis Monogioudis: Okay, that I want to map it into a price, right? So this one that will be solved by…
Pantelis Monogioudis: Taking the H of T as input.
Pantelis Monogioudis: Using now another projection, matrix V,
Pantelis Monogioudis: using another vector, C, And for me, In general.
Pantelis Monogioudis: I will write this in general, with a vector OT.
Pantelis Monogioudis: Is that, can be led for classification problems through a softmax.
Pantelis Monogioudis: to produce a YT hat.
Pantelis Monogioudis: Or, as in our case, can be itself led into a linear unit, with it, I suppose.
Pantelis Monogioudis: OT?
Pantelis Monogioudis: with a parameter vector E.
Pantelis Monogioudis: to form a wide cut.
Pantelis Monogioudis: T for regression.
Pantelis Monogioudis: And this is multi-lass.
Pantelis Monogioudis: classification.
Pantelis Monogioudis: I could have taken my H of D and used it to apply a linear unit straight away. Obviously, I can do that, right? But the reason, actually, I wrote it like this is I want to really use this, which I will call a head.
Pantelis Monogioudis: Generally, for either of the two problems that I'm dealing with, either classification or, regression problems.
Pantelis Monogioudis: So I know I have my price.
Pantelis Monogioudis: This… and this price was generated from considering, over time.
Pantelis Monogioudis: multiple latent factors, which is enclosed inside the range of hidden vectors. So, could you explain this diagram again? Yeah. So, I said, at the end of the day, I labored up to this moment in time to produce a vector hidden state.
Pantelis Monogioudis: Page of key, right? How can I… I need to match it, though, I need to predict the price.
Pantelis Monogioudis: So, I can do… Adopt product, straight away.
Pantelis Monogioudis: Problem solved. For this specific problem, problem is solved. I have a price.
Pantelis Monogioudis: I will train these obviously as well during my training process to predict the price. Having said that, I wanted to draw a very general kind of head.
Pantelis Monogioudis: Which will first do the dimension reduction down to, let's say, K dimensions.
Pantelis Monogioudis: which is my number of classes that I have for multi-class classification, then plug in the softmax to get a proper posterior probability distribution.
Pantelis Monogioudis: And, I will not obviously have that in that case.
Pantelis Monogioudis: But if I do that.
Pantelis Monogioudis: In my problem back… in my problem, I still have to get a price, so I will go from OT, I will go to a price via a simple dot product.
Pantelis Monogioudis: Yes, I'm doing just like I did it with,
Pantelis Monogioudis: CNNs, right, which I had at some point at the end of this kind of CNN layers, I had a number of fully connected layers.
Pantelis Monogioudis: I hover here once.
Pantelis Monogioudis: Again, that is doing just the linear…
Pantelis Monogioudis: combination. I'm getting an OT vector, which is capital K by 1 vector.
Pantelis Monogioudis: Where capital K is the number of classes.
Pantelis Monogioudis: As we will see, in language modeling, this capital K is going to be a pretty large number, obviously, because we are going to predict the next token out of 10,000 options.
Pantelis Monogioudis: So this guy, obviously, is a K by 1 parameter vector of E.
Pantelis Monogioudis: And this guy is of evidently K by 1.
Pantelis Monogioudis: Because soft marks is not changing the dimensionality of the input. It's a vector in, vector output function.
Pantelis Monogioudis: So what I will do now is I can actually go back.
Pantelis Monogioudis: And I can imagine now a situation such as this.
Pantelis Monogioudis: I'll draw it with two,
Pantelis Monogioudis: stages. In my attempt to answer the question, can't… Simple… RNNs.
Pantelis Monogioudis: Capture.
Pantelis Monogioudis: long, Darren.
Pantelis Monogioudis: dependencies.
Pantelis Monogioudis: Yep.
Pantelis Monogioudis: So I will design, two columns, right, and it's with a head now.
Pantelis Monogioudis: The first column will start with some kind of input at the bottom.
Pantelis Monogioudis: XD-1.
Pantelis Monogioudis: And the second column will be with the input XT, so it's just two stages in time to just show the principle of what I'm talking about here.
Pantelis Monogioudis: Okay, so obviously I have a gate.
Pantelis Monogioudis: which is, U, XT minus 1. Remember the U matrix from the later discussion, right?
Pantelis Monogioudis: U was the matrix that was processing the input XT-1. Of course, we concatenated 2, right, to, produce,
Pantelis Monogioudis: the W matrix.
Pantelis Monogioudis: So, here I have,
Pantelis Monogioudis: H.
Pantelis Monogioudis: T minus 2.
Pantelis Monogioudis: what I call this matrix, W prime, right?
Pantelis Monogioudis: W prime, HT minus 2, correct?
Pantelis Monogioudis: And, I added them up.
Pantelis Monogioudis: Together with…
Pantelis Monogioudis: A V vector. That was the equation of the single layer that I started the discussion. The orange equation… the orange equation is what I'm drawing right now.
Pantelis Monogioudis: I want to make a point, that's why I want to go back to the order's equation, not to straight away the concatenated equation.
Pantelis Monogioudis: I'm getting, effectively, what I called earlier as an activation at this moment in time.
Pantelis Monogioudis: Let's call this an activation, which is going to…
Pantelis Monogioudis: result into the hidden state, HT-1, After this vector, Go to the nonlinearity.
Pantelis Monogioudis: And obviously, I will have,
Pantelis Monogioudis: I will have, a tensor V.
Pantelis Monogioudis: That will multiply this kind of vector.
Pantelis Monogioudis: Add a corresponding bias kind of term.
Pantelis Monogioudis: And, heat it with a soft max.
Pantelis Monogioudis: to produce YT-1.
Pantelis Monogioudis: Correct.
Pantelis Monogioudis: That itself will go through Across entropy.
Pantelis Monogioudis: with a ground truth, YT minus 1, to produce a loss LT-1.
Pantelis Monogioudis: Okay, so in its full glory, what I have is the orange equation over here.
Pantelis Monogioudis: They can't… And, components, of…
Pantelis Monogioudis: classification which I need in order to produce a loss.
Pantelis Monogioudis: So let me know which one you cannot figure out from my excellent handwriting.
Pantelis Monogioudis: YT-1, cross entropy, So, Y hat minus 1, cosentropy.
Pantelis Monogioudis: YD minus 1, and LT minus 1.
Pantelis Monogioudis: Escape them.
Pantelis Monogioudis: I'm not going to draw the second stage. Yes, go ahead. So, if we're already doing tan H, why do we have to use forcemax at the end? TanH is the nonlinear function of the… it's like a webinar of random.
Pantelis Monogioudis: Even in regular, of course, we have to do the softmax in order to satisfy that is of the posterior probability distribution.
Pantelis Monogioudis: So, I'm gonna draw the second step, because the second layer in its full glory, because I'm claiming that anything
Pantelis Monogioudis: From this moment, This moment
Pantelis Monogioudis: To this moment, that is.
Pantelis Monogioudis: These are the same? Stay the same.
Pantelis Monogioudis: Okay, and obviously, I have here the tonnage.
Pantelis Monogioudis: manage.
Pantelis Monogioudis: I have an activation.
Pantelis Monogioudis: AT?
Pantelis Monogioudis: And then I have, B?
Pantelis Monogioudis: W H D.
Pantelis Monogioudis: HD-1.
Pantelis Monogioudis: Mr. W prime here.
Pantelis Monogioudis: Good evening.
Pantelis Monogioudis: HD-1.
Pantelis Monogioudis: And… There is some light at the end of a tunnel.
Pantelis Monogioudis: And it's not a trend. Okay, so here's the deal.
Pantelis Monogioudis: So, finally, we concluded. We plotted two instances. However, there is something missing from this diagram.
Pantelis Monogioudis: H of T minus 1 corresponds.
Pantelis Monogioudis: to the hidden state at T-1. Do you see another HC-1 here?
Pantelis Monogioudis: Over here.
Pantelis Monogioudis: Do you see that? So, I have.
Pantelis Monogioudis: a channel.
Pantelis Monogioudis: These two points are the same.
Pantelis Monogioudis: What's gonna happen now?
Pantelis Monogioudis: At some point, at the end.
Pantelis Monogioudis: In this specific case, I am going to have, in my price prediction problem, I will have
Pantelis Monogioudis: To do a prediction, therefore.
Pantelis Monogioudis: I will, create a loss at that specific moment in time, right? That… Los.
Pantelis Monogioudis: The gradient of this kind of loss with respect to the parameters, right, will be back propagated.
Pantelis Monogioudis: to all previous stages, right? The question I want to answer is, what Is the determinant factor of…
Pantelis Monogioudis: Problems that may be arising from that back propagation, which we'll call back propagation through time, but the name is not important.
Pantelis Monogioudis: Right? So, some tensor over here will play a very critical role into determining the gradient flow. Remember.
Pantelis Monogioudis: What's happening, which is key.
Pantelis Monogioudis: The contribution of the… a specific preview stage. Let us… let me plug another stage over here.
Pantelis Monogioudis: to the output.
Pantelis Monogioudis: Of my prediction will, of course, depend on how much gradient this guy gets.
Pantelis Monogioudis: Because if it doesn't get any gradient, then it doesn't change its parameters.
Pantelis Monogioudis: It doesn't change the parameters, then the hidden state is not updated.
Pantelis Monogioudis: Okay, so that's the most important, critical thing to think intuitively, right? So we'll see now the answer to the question, who is to blame? There's always someone to blame.
Pantelis Monogioudis: I told you the three principles of… did I tell you in this class? The three principles of corporate life in America?
Pantelis Monogioudis: Did I tell you about this? No? Okay. Just because it's Halloween, I will risk the joke. So what is the… what is the three principles of corporate life in the United States?
Pantelis Monogioudis: the first…
Pantelis Monogioudis: After 20 years of experience, 25 years' experience, I can tell you that's exactly 100% verified that it's true. Land grabbing.
Pantelis Monogioudis: Land grabbing is the first thing that happens in many corporations. I will be responsible for this, I will take your people, work in my department, blah blah, and so on, I'm…
Pantelis Monogioudis: You know, anyway, the man.
Pantelis Monogioudis: Import. So, the second principle is the principle of under-delivery.
Pantelis Monogioudis: Okay? This means that, most of the time, 99% of the time, the person who says, I am the man, under-delivers, okay? Or doesn't deliver at all what they're promising.
Pantelis Monogioudis: And the third principle, which is the most, the people, the one that they, also people see, is passing the buck.
Pantelis Monogioudis: So…
Pantelis Monogioudis: they… after they're under delivery, what's gonna happen? Someone has to be blamed, okay? So, this is happening in, domains of corporate life, politics, whatever. It's part of the human nature. Someone has to be blamed here about why this guy
Pantelis Monogioudis: Hasn't seen any gradient. And, do you know what tensor here?
Pantelis Monogioudis: If you follow the path, I'm draw… I'm drawing another path.
Pantelis Monogioudis: Right? From this kind of gradient, right? Backward variation through these stages, we have done it 100 times, right? You know, so nothing changes here. So…
Pantelis Monogioudis: Some gradients will be going through the tanage. Over here, there will be a summer.
Pantelis Monogioudis: summation gate. Following the, similar kind of thinking.
Pantelis Monogioudis: Notice this sentencer.
Pantelis Monogioudis: This tensor is… if you see the equation of backward navigation, you will see this tensor being a very critical tensor for the fate of the gradient.
Pantelis Monogioudis: Do we have control of this in this answer? We don't.
Pantelis Monogioudis: So, however, given that it has some kind of value, right, the dimensions of which you can see in your notes, right?
Pantelis Monogioudis: The gradient that arrives here We'll go here, correct?
Pantelis Monogioudis: Do you remember what was happening?
Pantelis Monogioudis: In these innocent traction gates.
Pantelis Monogioudis: The backward valuation summation.
Pantelis Monogioudis: So the gradient from the top now is going to arrive here, and it's going to sum to the gradient right over here.
Pantelis Monogioudis: to fall exactly the same trajectory to the previous guy, the previous guy who wants to play. Which, of course, brings us to the following observation. If you have a problem that you make a prediction only at the end.
Pantelis Monogioudis: That's even a worse situation, because why?
Pantelis Monogioudis: Because… You're not listening.
Pantelis Monogioudis: you don't have a gradient coming from above, because there's no need to actually form a loss. Like, for example, if I have a…
Pantelis Monogioudis: So, if I have a…
Pantelis Monogioudis: if I take a tweet, let's say as input, and someone tells me, is this a positive, negative, toxic, or whatever that is, right?
Pantelis Monogioudis: I will not…
Pantelis Monogioudis: producing any predictions before I am able to absorb the whole text for the tweet, right? So, I will be… only at the end, I'm going to determine, at the end of the day, what it is, and I will classify it. Which means that only at the end, in many problems, we have only at the end predictions.
Pantelis Monogioudis: In some other problems, we have predictions as we go.
Pantelis Monogioudis: Right? And therefore, predictions as we go, it's also, the case where I have a language model. Because if you think about it, I'm making the prediction now based on the previous kind of context, but then I'm going to use the…
Pantelis Monogioudis: previous context shifted by one to make a prediction of the next token and things like that, right? So, in some problems, we are assisted by these gradients at every instance in time. In some others, we don't.
Pantelis Monogioudis: It can be shown, and it has been shown, that,
Pantelis Monogioudis: Depending on the ideal decomposition, and the values of the
Pantelis Monogioudis: Of course, this W2' matrix is, if I remember right, this W' matrix is a neurons by a neurons, so it's a symmetric kind of matrix, right? It is very easy to hit it with an IG
Pantelis Monogioudis: the composition over there, in Python or MATLAB or whatever, and, find out, its, eigenvalues.
Pantelis Monogioudis: And, it has been shown that,
Pantelis Monogioudis: If the eigenvalues… so, if I'm writing… Beautiful.
Pantelis Monogioudis: the eigen… values.
Pantelis Monogioudis: Oh, W prime, is less than 1, Then the gradient…
Pantelis Monogioudis: is diminishing.
Pantelis Monogioudis: Here's the eigenvalues.
Pantelis Monogioudis: of W prime lower structure on the one.
Pantelis Monogioudis: Then?
Pantelis Monogioudis: gradient.
Pantelis Monogioudis: is exploding.
Pantelis Monogioudis: So we have two different scenarios. Either we have a diminished gradient or explosive gradient.
Pantelis Monogioudis: And when you have an exploding kind of gradient.
Pantelis Monogioudis: don't think that the situation is okay. Yeah, there will be some… lots of changes in that previous kind of stage, and most likely, the changes because of the representation in terms of dynamic range of our tensors, right, will probably see some form of clipping, and things will not work out.
Pantelis Monogioudis: as predicted, so I have to have a control of
Pantelis Monogioudis: Which, of course, brings us to another discussion I think we had, and if we didn't, we will have, on…
Pantelis Monogioudis: Cardo control… I think we had a discussion of batch normalization, right? Did we have a discussion of batch normalization at some point? Yes, people…
Pantelis Monogioudis: People already forgotten. So yes, we have… so someone has to control these tensors, right? And are we going to control the tensors, or something else? We will see in a moment, because that is going to be what the problem, the solution.
Pantelis Monogioudis: That was found that enhanced this temporal RNA to make it, what we call LSDM.
Pantelis Monogioudis: long, short-term memory at work, which is a topic that in itself may last for one lecture.
Pantelis Monogioudis: By itself, so I will compress it to 10 minutes.
Pantelis Monogioudis: Okay, because we are not going to, finally adopt it for what we are going to, do, LSDN. Having said that, it is, very interesting to see, to understand.
Pantelis Monogioudis: What they have done to avoid the situation.
Pantelis Monogioudis: Okay, if you were to say some lots.
Pantelis Monogioudis: of this thing, or if you want to get this kind of intuition. In your notes, I have,
Pantelis Monogioudis: your core side, Let me see, I think I have here.
Pantelis Monogioudis: I can go from here.
Pantelis Monogioudis: your core site, I have,
Pantelis Monogioudis: A scalar version of what is going on here.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: So here, at some point.
Pantelis Monogioudis: I have, as simple as possible.
Pantelis Monogioudis: Remember the single neuron? The current kind of neuron? There was some W scalar that was multiplying HT-1. Now this W scalar became W-ground matrix, right? If the W scalar is…
Pantelis Monogioudis: Let's say, minus 0.9.
Pantelis Monogioudis: And you start with an initial state of H0,
Pantelis Monogioudis: Guess what will happen through H of T? It will…
Pantelis Monogioudis: diminish, right? So the envelope of the…
Pantelis Monogioudis: of what is being produced, yes, it's diminishing. The enveloping the absolute value is diminishing, right? If it is larger than 1, the…
Pantelis Monogioudis: It will explore.
Pantelis Monogioudis: So there is always some kind of simplest possible abstraction you can have in your mind to remember what is going to happen, especially with this kind of matrices, obviously.
Pantelis Monogioudis: That's a kind of a…
Pantelis Monogioudis: Abstraction of what is really happening?
Pantelis Monogioudis: And of course, I am going to now discuss
Pantelis Monogioudis: How they solve this kind of problem.
Pantelis Monogioudis: And the solution to this problem is called LSTN architecture.
Pantelis Monogioudis: Norm?
Pantelis Monogioudis: Not bad.
Pantelis Monogioudis: memory.
Pantelis Monogioudis: So I'm going to draw just two diagrams, two stages in time, HT minus 2. Now, this would be left to right.
Pantelis Monogioudis: concatenation.
Pantelis Monogioudis: with H… sorry, xt minus 1.
Pantelis Monogioudis: the Mark XW.
Pantelis Monogioudis: We… H.
Pantelis Monogioudis: HD-1.
Pantelis Monogioudis: Conquer the nation now with… XD.
Pantelis Monogioudis: W.
Pantelis Monogioudis: B.
Pantelis Monogioudis: Don H.
Pantelis Monogioudis: H of P. This is basically the…
Pantelis Monogioudis: what I'll call the simple RNN cell, so I will take the simple RNN cell that I see over here, I will modify it, to… and draw it under it in order to understand the modifications. Okay, so…
Pantelis Monogioudis: I will introduce
Pantelis Monogioudis: long-term memory, in other words, a long-term hidden state, and I will maintain this H of T, as I will call it.
Pantelis Monogioudis: That's kind of a short-term heated state.
Pantelis Monogioudis: So I will have now a new letter that I will call S of T to… because I need a letter to represent another vector, right? That I call long-term state. So my job is to really understand
Pantelis Monogioudis: first of all, the diagram is very confusing, I should warn you, but we need to at least see how the long-term and the short-term states are
Pantelis Monogioudis: kind of trying to be maintained with the so-called gates, okay, that are doing such things here. So, right now, the… in the last name architecture, HT-2 is coming in.
Pantelis Monogioudis: And it goes through.
Pantelis Monogioudis: an input…
Pantelis Monogioudis: dense layer.
Pantelis Monogioudis: Together with HC-1.
Pantelis Monogioudis: And, an input gate.
Pantelis Monogioudis: His position over here.
Pantelis Monogioudis: That is going to also as input HD-2 and XT-1.
Pantelis Monogioudis: ST-1, which is a new long-term state, heathen State.
Pantelis Monogioudis: Is, going to go through NH?
Pantelis Monogioudis: And the output… is going to be called HT-1.
Pantelis Monogioudis: That is… going through.
Pantelis Monogioudis: Itself an output.
Pantelis Monogioudis: gate?
Pantelis Monogioudis: with input, Bing.
Pantelis Monogioudis: the usual two, values, HT minus 2, And XD-1.
Pantelis Monogioudis: And there's gonna be…
Pantelis Monogioudis: a highway.
Pantelis Monogioudis: Lying over there.
Pantelis Monogioudis: Which is only going to have a multiplier.
Pantelis Monogioudis: in it.
Pantelis Monogioudis: That is fed by the so-called forget gate.
Pantelis Monogioudis: Which, you guessed that it will be also be driven by XT-1 and HT-2.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: So this guy, what is, happening over here is,
Pantelis Monogioudis: what is… I'm lifting stuff out of this highway.
Pantelis Monogioudis: at time instance, T minus 2.
Pantelis Monogioudis: I'm adding it with the input.
Pantelis Monogioudis: a time instance.
Pantelis Monogioudis: T minus 1, the results of the input gate multiplication, and then I'm storing back into the highway Oops.
Pantelis Monogioudis: SD-1.
Pantelis Monogioudis: So what they have… basically, the… this is the end of the diagram, so I'll wait one minute, we'll finish rolling.
Pantelis Monogioudis: Okay, so what's happening now in a very, I would call it a compressed presentation of this kind of architecture is, I have these kind of gates.
Pantelis Monogioudis: That always result
Pantelis Monogioudis: between sigmoidal output units. They contain sigmoidal output units, and the sigmoidal output unit, basically, is applied, obviously, element-wise, and obviously results into numbers between 0 and 1. Because of that.
Pantelis Monogioudis: Whatever comes on the other part of the multiplier is multiplied with zeros and to one kind of numbers. So, effectively, what they do is, okay, some of this stuff that comes at the input of this, at the output of this input layer.
Pantelis Monogioudis: The input gate modulates and selects, effectively, softly, what to keep and what to forget.
Pantelis Monogioudis: Because it multiplies them with… Numbers between 0 and 1.
Pantelis Monogioudis: Obviously, we do not know what is the best value to use 0.1, 0.5, 0.8, 0.9, we do not know. Obviously, this will be determined
Pantelis Monogioudis: Because the input gate has trainable parameters during the training process.
Pantelis Monogioudis: The same thing happens with all other kids.
Pantelis Monogioudis: The output gate determines how much of this H of T we will propagate to the next step, because there's another step that is actually not drawn over here, which is a mirroring this, what we call the LSTM cell. So the LSTM cell…
Pantelis Monogioudis: As two inputs.
Pantelis Monogioudis: And two outputs.
Pantelis Monogioudis: The input is… H and S.
Pantelis Monogioudis: Alright, so S is living in,
Pantelis Monogioudis: In this kind of highway that we built, Purposefully, fully, to… allow its…
Pantelis Monogioudis: Propagation without going through a lot of… Stuff.
Pantelis Monogioudis: We have seen this before in the case of ResNets, which is also another instance of highway networks.
Pantelis Monogioudis: We had these skip connections, and if you remember, back then, we unrolled the diagram, and indeed, it was actually a line that was always from the input… from the output to the input in the dual background I use.
Pantelis Monogioudis: So… The only thing that… modulates this S of P is the forget gate.
Pantelis Monogioudis: how much of this kind of long-term hidden state I will be
Pantelis Monogioudis: remembering, and how much I would be forgetting. That's the job of the forget gate.
Pantelis Monogioudis: Okay, the other two gates are with respect to the input, and with respect to the short-term memory.
Pantelis Monogioudis: To give you an idea, if you have a task which is
Pantelis Monogioudis: summarization task, let's say, in actual language. And the input come soon.
Pantelis Monogioudis: Let's say, as the bank, Debunk again.
Pantelis Monogioudis: introduced.
Pantelis Monogioudis: And you… Savings.
Pantelis Monogioudis: account.
Pantelis Monogioudis: I'm sorry, I read the wrong line. That is the output of summarization. That is the output.
Pantelis Monogioudis: What is the input?
Pantelis Monogioudis: Of summarization is this guy.
Pantelis Monogioudis: The bank?
Pantelis Monogioudis: coma faced.
Pantelis Monogioudis: with… Pretty decent.
Pantelis Monogioudis: Over… Interest… rates.
Pantelis Monogioudis: introduced.
Pantelis Monogioudis: annual… Savings.
Pantelis Monogioudis: account.
Pantelis Monogioudis: Okay? That is the input.
Pantelis Monogioudis: In a task which is a summarization kind of task, the network
Pantelis Monogioudis: Should be… should learn how to not produce anything between the two commas.
Pantelis Monogioudis: the face with political over interest rates is not really necessarily carrying the final meaning of the essence of what I'm trying to say here. The fact is, I'm introducing a new account over here, right? So, with respect to this kind of input.
Pantelis Monogioudis: The input gate will be trained to not allow
Pantelis Monogioudis: This propagation of this kind of input tokens to affect either the long-term memory or the shorter-term memory.
Pantelis Monogioudis: And more specifically, H of T here, we see here. But I may need to remember the long term, but not necessarily
Pantelis Monogioudis: What is the… text between the commas. That's an example to…
Pantelis Monogioudis: Have in your mind, if you want to present to someone, you know, some intuition behind what are the functions of these gates are doing.
Pantelis Monogioudis: We don't have time to stay long here. However, I have a suggestion.
Pantelis Monogioudis: And, if you want to see some kind of animations of what's really happening inside an LSDM architecture.
Pantelis Monogioudis: in, Reference number 4.
Pantelis Monogioudis: Titled, Simplest.
Pantelis Monogioudis: possible SDM explanation video.
Pantelis Monogioudis: You will see an animated video of what's really happening in these kind of, gates.
Pantelis Monogioudis: When the task is to…
Pantelis Monogioudis: To predict what we are going to eat.
Pantelis Monogioudis: Tomorrow.
Pantelis Monogioudis: Which is…
Pantelis Monogioudis: Or what we're gonna cook tomorrow, which is an ancient worry for many families, and given the history of what we ate in the previous days.
Pantelis Monogioudis: Lots of animations there, I suggest you open up the video and watch it, just to understand a little bit what's happening. I don't think that you would be facing an LSTM question in the final.
Pantelis Monogioudis: Because we didn't really say that much, we just drew a diagram, and attempting to sort of suggest why this LSTM architecture is solving, if you like, this kind of gradient problem, because it attempts
Pantelis Monogioudis: To control certain things inside its gradient propagation.
Pantelis Monogioudis: it, it has this kind of highway where gradient is able to be provocated, remembered, and put back, ST-2 retrieved.
Pantelis Monogioudis: And, is added with the… whatever the input is to produce a new long-term memory ST-1 that is going to be put back into the highway, right? And then so on and so on, right? So,
Pantelis Monogioudis: That is… that is the… a lot of details we skipped about the equations. The equations are actually here. As you can see, all the gates have a sigmoidal unit at the input.
Pantelis Monogioudis: And, and you can actually read a little bit more about this.
Pantelis Monogioudis: Okay, so…
Pantelis Monogioudis: Let's now move on to the task that we were supposed to be doing, which is the…
Pantelis Monogioudis: So before we go into the transformer kind of architecture.
Pantelis Monogioudis: For regard to the transform architecture.
Pantelis Monogioudis: Oops.
Pantelis Monogioudis: I want to present a couple of things that we will see in front of us as tasks.
Pantelis Monogioudis: The first is, kind of the plain…
Pantelis Monogioudis: The plain old, task of,
Pantelis Monogioudis: The language modeling, the sort of language modeling.
Pantelis Monogioudis: Weird.
Pantelis Monogioudis: Alright, and then.
Pantelis Monogioudis: now we have the RNN, so we can actually do the… implement this kind of, P model that I was talking about, some time ago. So, effectively, the simple RNN, language model will…
Pantelis Monogioudis: take, W, D minus N.
Pantelis Monogioudis: We'll pass it through an embedding kind of layer.
Pantelis Monogioudis: but using the X D minus N.
Pantelis Monogioudis: Passing it through.
Pantelis Monogioudis: LLN slash LSTN, I'm not going to write LSTN, I just have in my notes, RNN, right? So, obviously, LSTN was used, but I have it here as RNN.
Pantelis Monogioudis: That is going to receive the hidden state H, T minus N.
Pantelis Monogioudis: Minus 1.
Pantelis Monogioudis: And to produce a hidden state age, Do you understand?
Pantelis Monogioudis: And then, with the appropriate kind of head, We produce a white hat.
Pantelis Monogioudis: of T minus N plus 1.
Pantelis Monogioudis: So, another stage will follow. That's an unrolled kind of architecture. This is XT minus N plus 1.
Pantelis Monogioudis: We'll go through the embedding.
Pantelis Monogioudis: Discussion over here, the layer.
Pantelis Monogioudis: with the input being WT minus N plus 1.
Pantelis Monogioudis: with the appropriate kind of head, which we discussed about. This head will be now a classification head, will give me another Y hat.
Pantelis Monogioudis: T minus N.
Pantelis Monogioudis: Clash 2.
Pantelis Monogioudis: And this thing will continue.
Pantelis Monogioudis: Okay, and finally, in the last stage, I will be producing
Pantelis Monogioudis: Y hats D.
Pantelis Monogioudis: So this architecture, for example, will be… Able to do this.
Pantelis Monogioudis: Where is it? Okay. Over here.
Pantelis Monogioudis: It's exactly the same thing. Notation is slightly kind of different. At some point, I go with some figures or whatever.
Pantelis Monogioudis: some other polls. So, over here, the…
Pantelis Monogioudis: the White House team that I wrote from over there, is going to be… The probability of
Pantelis Monogioudis: the final kind of, next kind of token, given the previous kind of context, right? So…
Pantelis Monogioudis: Obviously, there would be a posterior across the whole vocabulary, so it would be a V-dimensional kind of posterior, and obviously there would be some, some probabilities that determine out of the softmax. In this case.
Pantelis Monogioudis: the next token will be, books, and so on. But not… not… I mean, sometimes we do not produce as the next token that wins the competition out of the softmax. I will present now an algorithm that is called the maximum likelihood Sequence Estimation, that is actually
Pantelis Monogioudis: determines, finally, what,
Pantelis Monogioudis: what is being produced. But, for now, for this specific simple kind of,
Pantelis Monogioudis: representation of a language model, and that is basically what is going to be produced, okay? So,
Pantelis Monogioudis: That is,
Pantelis Monogioudis: There is a notebook.
Pantelis Monogioudis: From, there's the character level RNN over here.
Pantelis Monogioudis: I think you should go through it.
Pantelis Monogioudis: It ha- because the character level RNN, they're… Tokenization is…
Pantelis Monogioudis: You know, the vocabulary is very small.
Pantelis Monogioudis: So, and you can actually see the…
Pantelis Monogioudis: Interesting, there's a figure missing here. And, and there is, formatting is completely screwed up. I will change that.
Pantelis Monogioudis: However, not during the weekend, because I don't have access to my laptop, right? I'm traveling. So, I'll change that. But, here I'm actually, going through first principles.
Pantelis Monogioudis: Of, defining all the back propagation equations.
Pantelis Monogioudis: for all the tensors involved, for just a plain old RNN network, right? That effectively produces the next character, right, in the sequence, right? So here you can actually see what's happening. Obviously, this is a by-hand limitation of the equations, which you cannot read right now, because they have not been formatted properly.
Pantelis Monogioudis: And,
Pantelis Monogioudis: And then, evidently, in the training kind of process, we are starting from some, garbage over here, right?
Pantelis Monogioudis: And then, as the training kind of progresses, you will see that the RNN manages to predict successfully the next character. How we know that? Because, obviously, it is, where…
Pantelis Monogioudis: We are feeding the actual text to be predicted, so we know, we know it's a paragraph in which we do.
Pantelis Monogioudis: This, this code, use stuff.
Pantelis Monogioudis: It's a famous quote, because it was the first code that showed,
Pantelis Monogioudis: to students how to treat simple language models written by Andrei Karpathy when he was in Stanford decades ago.
Pantelis Monogioudis: Alright, so, with that kind of behind us.
Pantelis Monogioudis: What I want to do now is I want to, present,
Pantelis Monogioudis: a task which is called neural machine translation, so two tasks for the kind of a similar end. So, neural machine translation is the next kind of task.
Pantelis Monogioudis: Which is also an example of a sequence-to-sequence model.
Pantelis Monogioudis: In your translation, I have some sequence in one language, and I will get another sequence in a different language.
Pantelis Monogioudis: So I just want to see… I want you to understand what is really happening. Again, the whole discussion is, kind of on the superficial side, in the sense that we will see some block diagram in front of us, but I will describe in this kind of block diagram what's going on.
Pantelis Monogioudis: So, I have some input language, which is foreigner. It's a foreigner language.
Pantelis Monogioudis: And also, we have each translation.
Pantelis Monogioudis: I will be using.
Pantelis Monogioudis: X1, X2, and X3 as the tokens of the input language. I will do word-level tokenization for simplicity.
Pantelis Monogioudis: And for understanding, and I will denote as X prime, X2', and X3 prime the output tokens.
Pantelis Monogioudis: And, the neural machine translation will involve An encoder, and a decoder.
Pantelis Monogioudis: Okay, the job of the encoder is to look at the input language, and produce
Pantelis Monogioudis: a hidden state at the end of this encoding process for the last token, and again, in our backup formats, we always need to think about this encoder as this unlocked architecture that involves, in this case, three stages, because there are only three tokens. So the encoder will start its life with some
Pantelis Monogioudis: are an analyst DM, it doesn't really matter. With some input an initial kind of hidden state, it will form
Pantelis Monogioudis: H1.
Pantelis Monogioudis: when the… X2 matrix is coming in, it will form,
Pantelis Monogioudis: an H2?
Pantelis Monogioudis: And finally, In this trivial example, it will form And… H3.
Pantelis Monogioudis: And this H3 will be called
Pantelis Monogioudis: the thought vector.
Pantelis Monogioudis: I'll show you next one.
Pantelis Monogioudis: Why we are treating,
Pantelis Monogioudis: Why we are feeding the input tokies in kind of reverse order, that's a trick.
Pantelis Monogioudis: to buy us something, I will explain why, right? Because I didn't fit X1, I had X3, and then in kind of a reverse order, right? Another trick that they have played in this kind of architecture is the fact that, you know, these RNNs are bi-directional.
Pantelis Monogioudis: Bidirectionality also buys us something that I will also explain, but I would just want to finish the diagram for now. This kind of thought vector is the only which is many people represent
Pantelis Monogioudis: As the coatination of the forward and backward.
Pantelis Monogioudis: 5S…
Pantelis Monogioudis: by R.
Pantelis Monogioudis: sort of vectors.
Pantelis Monogioudis: These vectors, in other words, are fed.
Pantelis Monogioudis: this vector Fi, is fed as input to the decoder.
Pantelis Monogioudis: So this is the encoder.
Pantelis Monogioudis: To get to the phi vector.
Pantelis Monogioudis: And, here is a decomure.
Pantelis Monogioudis: This is the decoderacy report.
Pantelis Monogioudis: Okay, so what happened? What's going on here? Many things going on at the same time? First, you probably noticed the incorporation of some spatial tokens called the start of sentence and end of sentence.
Pantelis Monogioudis: Where I surrounded many instances my text to allow me to detect whether that
Pantelis Monogioudis: The decoding kind of process can start, and the decoding process can slow, right?
Pantelis Monogioudis: And because I'm decoding, I'm… when I'm feeding the start of sentence, I will predict by it.
Pantelis Monogioudis: when i comes in as input, because X1 prime is I, right, I'll predict laugh.
Pantelis Monogioudis: And when, love is the input, I'll predict you.
Pantelis Monogioudis: And when U is an input on predicted end of sentence.
Pantelis Monogioudis: And these kind of tokens are used also as indicators as to whether I'm going to do certain resetting
Pantelis Monogioudis: ingredients in cloud propagation procedures that are going on in this training, end-to-end, right? So I will explain what's happening now end-to-end, because all of these predictions, right, during training.
Pantelis Monogioudis: are going to be, evidently lead to losses, to classification kind of losses, the same heads that we've seen in a multi-class classification with the softmax and things like that up to this moment in time.
Pantelis Monogioudis: But, during an inference, I don't have training data, so what's going to happen during inference? I'm going to be feeding, effectively, the prediction.
Pantelis Monogioudis: I'm going to be fedding… feeding the predictions.
Pantelis Monogioudis: Into the next stage.
Pantelis Monogioudis: So the blue arrows are only during incidents.
Pantelis Monogioudis: Because I don't have training data. I don't have ground truth.
Pantelis Monogioudis: This is, Called Teacher Forcing.
Pantelis Monogioudis: There's always a very, complicated term.
Pantelis Monogioudis: In, this domain to describe something which is, too simple.
Pantelis Monogioudis: So, picture forcing effectively means, that during Raining?
Pantelis Monogioudis: we feed…
Pantelis Monogioudis: the ground truth.
Pantelis Monogioudis: VR.
Pantelis Monogioudis: The decoder inputs.
Pantelis Monogioudis: while inference, While during inference, We feed.
Pantelis Monogioudis: the previews.
Pantelis Monogioudis: Predicted token?
Pantelis Monogioudis: at angles.
Pantelis Monogioudis: That's kind of point number one that finishes. I mean, this is… this, this comment is there to finish, effectively, the flow of information that you see here on the neural machine translation system, but it does not, yet, explain why, what the reverse order
Pantelis Monogioudis: Of the tokens that you see here, bias.
Pantelis Monogioudis: this was a trick that it is, helped a lot, certain kind of, languages, because many languages, because we're dealing with translation over here, but not only, this is a general sequence-to-SQL kind of model, are starting
Pantelis Monogioudis: with, the subject… Followed by examiner.
Pantelis Monogioudis: Followed by object, right? So, effectively, in certain… in many grammatical kind of patterns, we see, especially in English, obviously other languages do not have this grammatical pattern.
Pantelis Monogioudis: The very first
Pantelis Monogioudis: thing from their source language is also the very first thing from the target language. So by effectively presenting
Pantelis Monogioudis: this token as the last input over here. What we're trying to do is we're trying to get a good footing of what is going to be produced first.
Pantelis Monogioudis: Awesome.
Pantelis Monogioudis: Because this thing is recent, right? So in terms of, propagation of the Indians, right, it will actually help us.
Pantelis Monogioudis: get this kind of reason information. That is the… explains the reverse kind of order. The bi-directionality that you see in many kind of architectures, such as this, is explained from, looking at,
Pantelis Monogioudis: claiming, okay, and I want to give you an example to remember that. In many instances, the…
Pantelis Monogioudis: Meaning of a word is…
Pantelis Monogioudis: can be defined not only based on what came before it, but also, in essence, what came after it, right? So, I want to give you an example to, to remember that.
Pantelis Monogioudis: the troops… Where?
Pantelis Monogioudis: More young.
Pantelis Monogioudis: 2 o'clock.
Pantelis Monogioudis: George Washington.
Pantelis Monogioudis: George Washington, is closed.
Pantelis Monogioudis: to traffic.
Pantelis Monogioudis: Good day.
Pantelis Monogioudis: You cannot please report after the JFOs.
Pantelis Monogioudis: So this is an example where you can actually see, okay,
Pantelis Monogioudis: this, guy, this GW here is a person.
Pantelis Monogioudis: Well, this avenue here is a bridge.
Pantelis Monogioudis: Right, has a meaning of a structure.
Pantelis Monogioudis: That is obviously determined after we see what follows it, right? While this guy can be determined
Pantelis Monogioudis: After we see what proceeding.
Pantelis Monogioudis: So bidirectionality allowed us to get the third vector, which I called pi f
Pantelis Monogioudis: in the forward direction, and pi r in the reverse kind of direction. And if we got both of them together in the fourth vector that we concatenated to, then we are performing better.
Pantelis Monogioudis: Because we are able to accommodate these two kind of cases, such as this.
Pantelis Monogioudis: Another comment I want to make is… so this one explained, by the… by the direction.
Pantelis Monogioudis: Nality.
Pantelis Monogioudis: of, this architecture.
Pantelis Monogioudis: Another thing what's actually going on is that, as you can see here, the whole decoding process.
Pantelis Monogioudis: Depends on this guy.
Pantelis Monogioudis: As we'll see.
Pantelis Monogioudis: Definitely that we have improved today, because we are effectively… when we do neuroma simulation using transformers, then this is not happening. But back in… at the time, this was actually happening. One single vector was defining everything that we see here.
Pantelis Monogioudis: Obviously, the, during training, we were getting supervisory signals as well, obviously. But,
Pantelis Monogioudis: I want to make sure that you understand that this whole thing is jointly trained, so both encoder and decoder trainable parameters are jointly adapted.
Pantelis Monogioudis: So… the performance for…
Pantelis Monogioudis: I will call it prediction of the decoder, will definitely affect the representation that the encoder is sending us.
Pantelis Monogioudis: So this 5-fold vector is changing, right, over time, and to reflect that, because we're doing joint parameter optimization of both encoder and decoder.
Pantelis Monogioudis: And this is the last point I wanted to make.
Pantelis Monogioudis: I owe you, however, something that I told you, that it will come.
Pantelis Monogioudis: And, and this something is called, bill search.
Pantelis Monogioudis: That actually helped us to revisit previous determinations or predictions that we have, because…
Pantelis Monogioudis: We are going to be looking now at a sequence of predictions, rather than the individual memory-less, I will call it, or greedy prediction we were seeing in the softmax head a few moments ago. I'll explain that in a moment. This is also known as maximum
Pantelis Monogioudis: likelihood…
Pantelis Monogioudis: sequence.
Pantelis Monogioudis: destination.
Pantelis Monogioudis: Also known as a Viterbi algorithm from the inventor.
Pantelis Monogioudis: MLSE, the acronym that you will find is
Pantelis Monogioudis: algorithms in the literature as well. So this is basically what I would like to explain.
Pantelis Monogioudis: So, I will use the NMT, the neuromacentralization, as an example. Yes, go ahead. I had a question about training the bidirectional LST. Yes, yes.
Pantelis Monogioudis: how do we input the source language in the encoder if we are going in one place? Like, in bi-directional, we are going in the forward direction as well as the backward direction, right?
Pantelis Monogioudis: So at a particular time step, the input will be one token, right? It will be one token from the source language.
Pantelis Monogioudis: Yeah, so you're feeding the source language.
Pantelis Monogioudis: as X3, X2, X1, you predict a thought vector, you determine a thought vector, and then you're feeding it as X1, X2, X3, you determine another thought vector.
Pantelis Monogioudis: So, okay, so is the forward and the backward training happening simultaneously at each time step, or is it that we are first training it in the forward manner, then we are training it in a backward manner?
Pantelis Monogioudis: Okay, so, alright, so I think you've got a point there.
Pantelis Monogioudis: in the… I mean, the training-wise, I am not 100% sure about the details here, of how the training and protocol is going to be engineered, but obviously.
Pantelis Monogioudis: that combined 5-vector for whatever we are… I mean, we may be able to freeze, let's say, again, I need to verify that, may be able to freeze the weights, right, at some point in time, right? To say, okay, I have, in the previous kind of,
Pantelis Monogioudis: iteration, because this thing is telling us the source language coming one sentence after another. In the previous kind of iterations, I have left
Pantelis Monogioudis: my forward path, right, with these specific weights, let me start from that, and then feed it in X1, X2, and X3, right? And then I can go back and say, now I'm going to feed it also with X3, X2, and X1.
Pantelis Monogioudis: starting, for example, in the same way, to form the two thought vectors, the fourth and the reverse, right? With that book a negative file vector, I will run the decoding tosses.
Pantelis Monogioudis: Which is obviously going to… Back from a gate.
Pantelis Monogioudis: that, those trainable parameters that I have done there. I will… I can verify that, it's not that I am,
Pantelis Monogioudis: 100% sure of even what I'm saying, but this is, potentially a solution.
Pantelis Monogioudis: Okay, so what I wanted to, talk a little bit about is I will use the neuromassy translation as,
Pantelis Monogioudis: As an example. And remember that we were doing this
Pantelis Monogioudis: We're starting from a start-of-sentence kind of token.
Pantelis Monogioudis: And, let's assume that, We are using…
Pantelis Monogioudis: This is, this is, by the way,
Pantelis Monogioudis: kind of an inference, kind of algorithm, so we are using this during inference. Let's assume that the posterior probability was, such, that the next, prediction was V.
Pantelis Monogioudis: Then, we had, again, with probability 1, we predicted ship.
Pantelis Monogioudis: Then… We had the… Again, this is a very limiting example. Two probabilities, 0.6 and 0.4, effectively.
Pantelis Monogioudis: And, with 0.6, The token, was Haas, and 0.4, the token behind this, was sailed.
Pantelis Monogioudis: And, then… we had…
Pantelis Monogioudis: 0.55.
Pantelis Monogioudis: 0.45.
Pantelis Monogioudis: Don't.
Pantelis Monogioudis: sunk?
Pantelis Monogioudis: Here it was 0.9. 0.1.
Pantelis Monogioudis: Chrome?
Pantelis Monogioudis: Ed?
Pantelis Monogioudis: away.
Pantelis Monogioudis: So, effectively, we… we have,
Pantelis Monogioudis: With, with kind of a beam search,
Pantelis Monogioudis: What we're trying to do is we're trying to prove, with this kind of discussion, that the greedy approach… so, let me write it down…
Pantelis Monogioudis: the greedy.
Pantelis Monogioudis: approach.
Pantelis Monogioudis: Of always selecting if, like, the maximum posterior probability at every stage of the decoding kind of process, right?
Pantelis Monogioudis: to determine the prediction that we're going to feed to the next stage, and so on, during inference, right? May not be the best thing we can do. I mean, that's the premise of the looking out at the whole sequence of predictions, rather than the instantaneous prediction, right?
Pantelis Monogioudis: So, I'm writing now that the ready approach of selecting
Pantelis Monogioudis: the token?
Pantelis Monogioudis: with… the reason I call it grid is that we are using the… Max posterior.
Pantelis Monogioudis: probability.
Pantelis Monogioudis: Art.
Pantelis Monogioudis: Andy.
Pantelis Monogioudis: instance… In time.
Pantelis Monogioudis: Results.
Pantelis Monogioudis: In this, this, in this, sort of decoding.
Pantelis Monogioudis: the sheep?
Pantelis Monogioudis: Pass?
Pantelis Monogioudis: docked.
Pantelis Monogioudis: So, we have predicted By this specific thing, this trajectory.
Pantelis Monogioudis: Do you hear me?
Pantelis Monogioudis: This is basically the greedy approach. However, If you do the calculations.
Pantelis Monogioudis: Of, what is the calculation here?
Pantelis Monogioudis: The calculation of maximizing.
Pantelis Monogioudis: Amazing.
Pantelis Monogioudis: the likelihood
Pantelis Monogioudis: off… at sequence… of… translated tokens.
Pantelis Monogioudis: results.
Pantelis Monogioudis: with K is equal to 2.
Pantelis Monogioudis: Let me not bring up the K parameter yet, but let's look at this.
Pantelis Monogioudis: This one, will be, the maximum… the likelihood will be… The product of the probabilities.
Pantelis Monogioudis: Remember, not logged, that's just a likelihood. That is basically 0.33. 0.33 will be the…
Pantelis Monogioudis: Likelihood of this trajectory, right?
Pantelis Monogioudis: Another trajectory, however, will be… I don't have a calculations for all the trajectory, so another trajectory will be…
Pantelis Monogioudis: 0.36.
Pantelis Monogioudis: In fact, this is the one that is the largest, in terms of the likelihood of these sequences, right? Between, in other words, the two.
Pantelis Monogioudis: I will select, in this specific, after the application of this meme search algorithm, I will select the magenta one.
Pantelis Monogioudis: And so… so maxima… I'm writing that maximizing the sequence of a translated token does not
Pantelis Monogioudis: is not.
Pantelis Monogioudis: Sorry, what is not, is the optimal thing.
Pantelis Monogioudis: Again, according to the maximum criterion, anyway, of the sequence, because we have, let's say, a sequence-to-sequence problem over here, right? So, across this whole sequence of translations, I want to make sure that the maximum is really not at the individual kind of translation level, but it's across the whole sequence.
Pantelis Monogioudis: That's the best thing to do.
Pantelis Monogioudis: So, how we, how we do that, how we maintain, how we ended up with the magenta versus the green? So, the… the algorithm…
Pantelis Monogioudis: Maintains.
Pantelis Monogioudis: log probabilities.
Pantelis Monogioudis: Oof.
Pantelis Monogioudis: many branches.
Pantelis Monogioudis: And therefore, we are able to calculate
Pantelis Monogioudis: Not only the winning branch, but the second winning branch, the third winning branch.
Pantelis Monogioudis: And, capital G, obviously, the hyperparameters that we are going to push under the rug.
Pantelis Monogioudis: And, call it the day.
Pantelis Monogioudis: Yes. I'll provide a research, in the meeting?
Pantelis Monogioudis: Like, how down… like, how many levels do we search? Okay, so obviously, we are going to do… maintain that, for,
Pantelis Monogioudis: let's say, remember there was this kind of end-of-sentence token that we are going to predict? We are obviously going to maintain these branches until that end-of-sentence token. Then, of course, if we start a new translation, we will start from the beginning of sentence, or start of sentence, to, again, go into the end of sentence.
Pantelis Monogioudis: So, wouldn't that be extremely, like, computationally expensive compared to… Yeah, so, yeah, so as you can see here, even in this kind of simple example, you will actually get a quite significant
Pantelis Monogioudis: number of branches very, very quickly, right? So obviously, this capital K is a hyperparameter, is typically a small number. So here, you just see the situation with 2.
Pantelis Monogioudis: Obviously, the longer this thing lasts, like, this kind of translation, the more branches we can potentially need to track, right?
Pantelis Monogioudis: So, I know that the capital K is a hyperparameter. It is, ordered in the…
Pantelis Monogioudis: in the results that we're using, or it can be reconfigured in many user interfaces that I have seen.
Pantelis Monogioudis: So this was basically the basic principles of big cells.
Pantelis Monogioudis: Which, of course, is the…
Pantelis Monogioudis: Last thing I wanted to mention before going into… The final discussion on metrics.
Pantelis Monogioudis: NLP metrics. There are plenty of MLT metrics. I'm just going to mention what is called the blue metric.
Pantelis Monogioudis: So this is, again, using the example of neural machine translation.
Pantelis Monogioudis: So, in, in many kind of problems, like, you know, most intersection, we have some kind of an expert.
Pantelis Monogioudis: annotators.
Pantelis Monogioudis: That, we'll be… called to do the… to give us translation about the source language. So…
Pantelis Monogioudis: If the neuromachine translation predictor resulting into, let's say, the plane Blue.
Pantelis Monogioudis: in… Boom.
Pantelis Monogioudis: Athens, let's see.
Pantelis Monogioudis: The human translator will produce the plain Two cough.
Pantelis Monogioudis: from Athens.
Pantelis Monogioudis: And, the plane.
Pantelis Monogioudis: The other guy will say that Lane departed, Throw my offense.
Pantelis Monogioudis: I'll use this kind of translation kind of examples, with the ground truth.
Pantelis Monogioudis: To, tell you a little bit about the…
Pantelis Monogioudis: sort of the metric, which is, it's one of the many, but it's absolutely quite popular. It's called Blue.
Pantelis Monogioudis: So… measures… Similarity.
Pantelis Monogioudis: using… and gram.
Pantelis Monogioudis: overlaps.
Pantelis Monogioudis: So I'll give you an example, based on this kind of translation to understand. So let's just… let's focus a little bit on the diagrams.
Pantelis Monogioudis: Ultimately, blue will be linked to the precision as an equation, where precision metric as an equation, so…
Pantelis Monogioudis: Biograms. Okay.
Pantelis Monogioudis: What is going on here?
Pantelis Monogioudis: Scream corner.
Pantelis Monogioudis: iPhone, whatever.
Pantelis Monogioudis: Oh, no.
Pantelis Monogioudis: Potential spam. Okay.
Pantelis Monogioudis: Blue.
Pantelis Monogioudis: Alright, so… I have here the…
Pantelis Monogioudis: I'm using holy words to understand the plane.
Pantelis Monogioudis: Is the plane present?
Pantelis Monogioudis: in the… as a background, let's say, in both the… of the human… human translators, it is, so this one, I will call it as through post-de-event.
Pantelis Monogioudis: How about plane flume?
Pantelis Monogioudis: No.
Pantelis Monogioudis: That is false positive.
Pantelis Monogioudis: blew in.
Pantelis Monogioudis: Another false positive.
Pantelis Monogioudis: in from… and other false positives, I don't think we're doing very well. And, from Athens.
Pantelis Monogioudis: This is what?
Pantelis Monogioudis: A true positive.
Pantelis Monogioudis: So… Effectively, as true positives. We effectively categorize Ngrams.
Pantelis Monogioudis: in C, by the computer, in other words. So maybe the C is very confusing, so this is basically the machine.
Pantelis Monogioudis: And these are… are… Two months.
Pantelis Monogioudis: Okay, so, n-grams in C… That are also… in our book.
Pantelis Monogioudis: And what is a false positive event?
Pantelis Monogioudis: It's the negation of that statement, okay? I'm not going to write it. Okay, that's the false part of the event. So the… the blue has precision semantics.
Pantelis Monogioudis: It's a bit more complicated than what I'm describing here.
Pantelis Monogioudis: Almost the precision in true positive is divided by true positive plus false positives.
Pantelis Monogioudis: So this is basically, in this specific example, it will be 2 fifths.
Pantelis Monogioudis: But machines are also…
Pantelis Monogioudis: going to, able to game the system, right? In fact, I find there are many ways of gaming the system, so if I, translate something…
Pantelis Monogioudis: Okay, then, or, or…
Pantelis Monogioudis: the plane, the plane, the plane, or whatever that is, right? Obviously, My translation is completely wrong.
Pantelis Monogioudis: But, the metric is going to shoot up.
Pantelis Monogioudis: And, and, and…
Pantelis Monogioudis: the main formula, I mean, the final formula of precision penalizes about shortness of translations, or translation can receive in this form of repetition. There are… it's a bit more complicated as a formula, and your notes have the…
Pantelis Monogioudis: final formula that I… so I don't insist too much on, metrics, per se, although they are important for benchmarking.
Pantelis Monogioudis: at this moment in time, so I don't… I don't have my final kind of form, but you can go to the website and see the final forum and together some more examples.
Pantelis Monogioudis: The reason I'm kind of rushing is I want… I want to at least tell you something about Transformers today, because,
Pantelis Monogioudis: Which is basically the final destination with the final URL architecture we're going to see in this course.
Pantelis Monogioudis: So, I'm going to start the discussion, at least for the next 20 minutes, okay? So, translation… sorry, transformers.
Pantelis Monogioudis: So, we were going to be, concerned with, two things, architecturally, two things. The first thing, the first thing is that, we eliminate
Pantelis Monogioudis: the recurrent connections.
Pantelis Monogioudis: In other words, the need to form an H of T from HD-1.
Pantelis Monogioudis: So, effectively, we are converting a Syrian Oriented kind of architecture.
Pantelis Monogioudis: To a parallel architecture, where now, obviously, there are some kind of side effects from that.
Pantelis Monogioudis: Because in the serial kind of architecture, the HT-1 was presented to us from an input XT-1, and then HD was not…
Pantelis Monogioudis: available to us until XT-1, and the input XT is available to us. Look at the RNN kind of architecture over there. Now, if we don't have this kind of relationship, we have to manually inject.
Pantelis Monogioudis: the positions of the sequence of the targets, right? So we have to… this is the second step, the second kind of thing that we have to do. We need to inject
Pantelis Monogioudis: positional information.
Pantelis Monogioudis: with respect, to input.
Pantelis Monogioudis: token. Applicants.
Pantelis Monogioudis: Because we are feeding everything in parallel, so transformer architectures are permutation invariant structures, unless you do explicit accounting for that position of every token.
Pantelis Monogioudis: Fuck And the third and most important one.
Pantelis Monogioudis: This is kind of a sides thing, okay?
Pantelis Monogioudis: the number 2. But the most important one is that, we need to build.
Pantelis Monogioudis: Contextual.
Pantelis Monogioudis: abilities.
Pantelis Monogioudis: in a… In an approach that we'll call the attention mechanism.
Pantelis Monogioudis: So what is the contextual abilities? As we're discussing in the context 3 abdings, the contextual embeddings
Pantelis Monogioudis: will change position in the D-dimensional kind of space, depending on what other tokens are next to them.
Pantelis Monogioudis: In other words, that…
Pantelis Monogioudis: bank that I use as kind of an example, depending on the meaning of the term, will now be marked into two different positions, not something like an average of many.
Pantelis Monogioudis: So we'll see how this is actually done, using not bank, for a change as an example, but, the term, bears. Okay, so I have, let's say, 3 sentences, just to explain the principle behind it.
Pantelis Monogioudis: The first sentence is, I love.
Pantelis Monogioudis: Bears?
Pantelis Monogioudis: In the second sentence is, is seed bears.
Pantelis Monogioudis: the paint?
Pantelis Monogioudis: And the third sentence is, bears.
Pantelis Monogioudis: won the game.
Pantelis Monogioudis: So, obviously, the token pairs, which again, I'm using as a kind of a whole-world token for easiness of,
Pantelis Monogioudis: a sort of, description. So, this guy over here, is… Has a meaning of, the animal.
Pantelis Monogioudis: This guy over here has a meaning of… tolerance.
Pantelis Monogioudis: And,
Pantelis Monogioudis: I'm sure somewhere in the US, there would be a team called Bears. Sports team. Which city is this? I don't know, Chicago. I think I remember that from last time.
Pantelis Monogioudis: So this is the sports channel.
Pantelis Monogioudis: I know nothing about these links, but I heard that it is indeed somewhere.
Pantelis Monogioudis: Is it baseball, right? This is football, yeah. Football, right?
Pantelis Monogioudis: Alright, so,
Pantelis Monogioudis: So, the, the idea here is,
Pantelis Monogioudis: We are going to have a context.
Pantelis Monogioudis: In each kind of sentence. Some tokens that are appearing in the same sentence with the bears over there will push the bears
Pantelis Monogioudis: In some other place in the D-dimensional space.
Pantelis Monogioudis: And there will be 3 points in this three-dimensional space that bears will be pushed, okay?
Pantelis Monogioudis: And this is what the attention mechanism is achieving. I want you to think about the attention mechanism as this change in the D-dimensional space that we introduced.
Pantelis Monogioudis: So, I'm going to describe now the kind of simplest possible version of this attention mechanism, which is not implemented, but I'm just going to discuss it because it's really simple. So the…
Pantelis Monogioudis: It's kind of a stepping stone to understand when it is implemented. So, the untitled, the simplest, formed.
Pantelis Monogioudis: Of the attention mechanism.
Pantelis Monogioudis: So, I'm going to take, all this, kind of, tokens, and, I am going to store them in this kind of, matrix, capital X.
Pantelis Monogioudis: Okay, that is going to have dimensions.
Pantelis Monogioudis: D?
Pantelis Monogioudis: The well-known dimension that we have,
Pantelis Monogioudis: In terms of a betting space, right? We use this letter, if you remember. And, capital T,
Pantelis Monogioudis: Which, for the interest of this discussion, I'll set these numbers to very small numbers.
Pantelis Monogioudis: capital T3, and capital monitor D4. This capital, T will be the context size.
Pantelis Monogioudis: And you know what, D's…
Pantelis Monogioudis: The simplest possible attention mechanism is using a dot product, something that will be inherited as an operation for their more elaborate kind of attention mechanism a bit later, to do the following.
Pantelis Monogioudis: I will calculate a score matrix, S,
Pantelis Monogioudis: Which is, in this case, will be… P by T, right?
Pantelis Monogioudis: by doing the operation X, X times pose.
Pantelis Monogioudis: This work, obviously, will result into some matrix.
Pantelis Monogioudis: S11, S12, S13, finally S33.
Pantelis Monogioudis: For that specific example that we have, T is equal to 3.
Pantelis Monogioudis: Then I'm going to, first of all, why am I actually doing this kind of dirt product, right?
Pantelis Monogioudis: I mean, if you can imagine, the tokens already have gone through the stage of an embedding layer.
Pantelis Monogioudis: And therefore, they have mapped themselves somewhere, right? With this kind of dodge product, what I'm trying to do is to see
Pantelis Monogioudis: Is this kind of other token can offer something.
Pantelis Monogioudis: to the new location of the token bears. Is there just in tokens? Including the token itself of bears, right? So I'm doing what's called self-attention.
Pantelis Monogioudis: And I'll calculate these kind of dot products and see what will happen now. I will finish this, and you will see what is going to happen. I am going now to pass this kind of,
Pantelis Monogioudis: matrix through Softmax.
Pantelis Monogioudis: 12Max was used always in this course as a posterior probability distribution generator. Over here, we are not going to use it as such.
Pantelis Monogioudis: We are going to use it as a normalization device that is going to take this matrix, the matrix S row by row, and it will create the following.
Pantelis Monogioudis: We'll create another matrix.
Pantelis Monogioudis: which I'll call the attention weights. Again, obviously, 3 by 3.
Pantelis Monogioudis: So, this S came from scores.
Pantelis Monogioudis: And this A… Because of the…
Pantelis Monogioudis: Nature of the softmax matrix is… called Attention Weights.
Pantelis Monogioudis: Isomatic has stored the attention weights.
Pantelis Monogioudis: that obviously, because of softmax, I have the property. AIJ is greater than or equal to 0.
Pantelis Monogioudis: And the summation.
Pantelis Monogioudis: from J is equal to 1.
Pantelis Monogioudis: to capital T, AIJ is equal to 1.0.
Pantelis Monogioudis: So every role on this, kind of attention kind of matrix.
Pantelis Monogioudis: The summation of this attention weight for every row is 1.0.
Pantelis Monogioudis: what I have achieved up to this moment in time, I basically said, okay, you know,
Pantelis Monogioudis: I will be using this attention now weights, to do the following. Create.
Pantelis Monogioudis: what I call contextual embeddings.
Pantelis Monogioudis: Spoilers.
Pantelis Monogioudis: Xi hat. This is the notation I will use for my contextual betting for the iF token.
Pantelis Monogioudis: is… a linear combination.
Pantelis Monogioudis: Thanks, Jude.
Pantelis Monogioudis: Or in another matrix form, you can see it as Thanks, Scott?
Pantelis Monogioudis: Capital X hat is softmax.
Pantelis Monogioudis: of XX transpose.
Pantelis Monogioudis: X.
Pantelis Monogioudis: All capital.
Pantelis Monogioudis: What the personal question is telling us?
Pantelis Monogioudis: in the D-dimensional space, and this is basically what I want you to remember out of this kind of a discussion.
Pantelis Monogioudis: I started.
Pantelis Monogioudis: from… Allocation in space, in a D-dimensional space, right?
Pantelis Monogioudis: And I ended up…
Pantelis Monogioudis: in another location, And the… and the mechanism that pushed me
Pantelis Monogioudis: To that location is a self-attention mechanism.
Pantelis Monogioudis: This is what I want you to remember out of this discussion.
Pantelis Monogioudis: Okay, in the simple, like, impression mechanism, however, and this is the pretextual that will be discussed in the next kind of lecture.
Pantelis Monogioudis: We see two things.
Pantelis Monogioudis: Although I ended up somewhere, based on the weights that the other tokens offered to me to move me, right?
Pantelis Monogioudis: This mechanism… Did not involve anything that it is associated with
Pantelis Monogioudis: with training, or with learning, and learned the fact that I learned how to do context-free abilities, because all these attention weights are the results of where these other tokens were mapped.
Pantelis Monogioudis: by another externality, not something that I created here. In fact, they came from that…
Pantelis Monogioudis: The scheme that we've seen in the previous lecture and reviewed today.
Pantelis Monogioudis: the Word2VAC context-free embeddings.
Pantelis Monogioudis: This is a deterministic mapping, in other words, if I have these X vectors, this is where I will end up. In the next lecture, we will see and understand why this is not necessarily the best idea.
Pantelis Monogioudis: So why we… need to, do certain projections that we control.
Pantelis Monogioudis: And this will be the projections inside this attestion mechanism, will be the enhanced version of the attention mechanism.
Pantelis Monogioudis: that it will give us certain benefits in this kind of mapping. But I hope you understood what is the end result out of this attention, at least. How we will do it, more specifically, we'll do it next week.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: D.
Pantelis Monogioudis: Does anybody have that second sheet? Okay, I need to have page back weekly.
Pantelis Monogioudis: I mean, if it is a library that just wants Gibson, if you have a source code of Gibson, which we do, right?
Pantelis Monogioudis: This kind of source code, I think I have this source code is even in a framework called TensorFlow. Then you have to translate to Python, for example. If you are a TensorFlow person, though, you don't necessarily need to translate it.
Pantelis Monogioudis: And I would guide the PAs for… because, you know, it's an important thing, right? Is that where you're at? You're at the last part?