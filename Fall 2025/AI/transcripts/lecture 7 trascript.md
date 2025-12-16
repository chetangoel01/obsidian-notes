Pantelis Monogioudis: Okay, alright, let's get started.
Pantelis Monogioudis: please sign in here, do not forget, because…
Pantelis Monogioudis: Today, I have to leave at 1.20, and we need to… I need to take the paper with me at 1.20 sharpening.
Pantelis Monogioudis: Here it comes.
Pantelis Monogioudis: So, if you go to the Discord, under your, midterm, Friend?
Pantelis Monogioudis: There's something I want you to do.
Pantelis Monogioudis: Read what I'm saying and answer the question.
Pantelis Monogioudis: with your thumbs up and thumbs down. Do it right now. I'm curious to see how you did in your midterm. Yeah. Thumbs up means that you did as expected, and thumbs down you did not as expected, or below expectations.
Pantelis Monogioudis: There's no third category.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: I was very busy with drafting your project, and I am running one week late to give the solutions to the…
Pantelis Monogioudis: TAs. However, I think the TAs were… gave you something this week, right? Did they give you back something? Like an assignment grade? They did, they gave you something. If you missed it, yeah, it is there. All right, let's see the results. Did you vote?
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: Your vote is important because this will also determine what I'm going to say to the TAs with respect to lenience.
Pantelis Monogioudis: I'll be terminal and logistics. Okay, great.
Pantelis Monogioudis: Not a single thumbs up. Okay. All right. Yes, guys, please, please don't forget to sign in. I need to leave 120 today, so I kind of…
Pantelis Monogioudis: All right, let me tell you a little bit about the grade. How grades are working in this kind of class, so… so at least, some kind of a stress
Pantelis Monogioudis: is, sort of, subsides. All right, so let's assume that you, I mean, the midterm is only 20 points.
Pantelis Monogioudis: So I'm guessing that, you know, most of you got
Pantelis Monogioudis: something above 10. And therefore, and therefore, you know, you lost, let's say, 10 points. Then you did well. Very typically, the project, it's very typical to get most of the points, unless you did really badly. I'll come to the project in a moment.
Pantelis Monogioudis: And also in the… so what also remains is the final, okay, which is kind of 30 points. And, typically the final is cumulative, so we are testing everything. There are pros and cons about this. If it is not, and it's only after the midterm.
Pantelis Monogioudis: most of the material after the midterm is typically more difficult than the first material up until the midterm, right? So, most of the questions, if it was not like this, would have been from that more difficult kind of section, now it's going to be a balanced thing.
Pantelis Monogioudis: like your media. And, so, so given your vote, I will, mention something to the TAs to make sure that, at least
Pantelis Monogioudis: you know, your, we obviously need to be fair. The exam wasn't on the difficult side, I'm gathering, right? Was it on the difficult side? Okay. Yeah, alright, that's fine. So we'll, we'll process that and, come back on the final with some, better situation.
Pantelis Monogioudis: Okay, so how the grade is working? Let's assume that you got some points at the end of the class.
Pantelis Monogioudis: The total number of points are going to be, thresholded, by me, based on some secret threshold.
Pantelis Monogioudis: Okay? So I'm going to get some kind of nice, skewed Gaussian distribution. Of A, Bs, and so on and so on, right? So the beans are, typically, over here, what we notice is that people get, typically, letter grades better than what they were expecting.
Pantelis Monogioudis: That is what typically, was, sort of suggested to us over the years from teaching this course. So, yes, you need to put the effort, especially in the second half, where the project is going to keep you busy. And this is what I'm going to start today with the project, project, sort of discussion.
Pantelis Monogioudis: And, so in the second half, we will go through, starting today with some kind of principles of natural language processing.
Pantelis Monogioudis: We're gonna do, logical reasoning after that, planning without interactions, and planning with interactions.
Pantelis Monogioudis: Okay, that is really the plan for that.
Pantelis Monogioudis: For this, for the… for the moving forward.
Pantelis Monogioudis: Okay, so, so what the project is, okay? So, basically, you are called to
Pantelis Monogioudis: construct an AI tutor. If you will watch the news, you know, Khan Academy is doing that, lots of other people or startups are in the process of actually doing something like that, already at MOC. And this tutor has a name, it's called Erica. All right, so,
Pantelis Monogioudis: For the full implementation, Erica will be able to talk, hear you.
Pantelis Monogioudis: And, also see you, right? But over here, we are going to limit ourselves to
Pantelis Monogioudis: Textual interactions. Okay, and when I say textual interactions, not entirely textual, I'll come to that at a moment. It would be some kind of a visual component as well.
Pantelis Monogioudis: Someone will start with a kind of system architecture. System architecture will have, obviously, a user interface, and in this user interface, you are free to select
Pantelis Monogioudis: UI, which is an existing UI, you can select, for example, Open Web UI.
Pantelis Monogioudis: you can select some other chat UI, even a very simple web UI that you can construct from HTML on your own. It doesn't have to be something elaborate.
Pantelis Monogioudis: But definitely the UI should have some space where you can insert your prompts, and obviously we'll have some kind of probability to display the suggestions, right? The responses.
Pantelis Monogioudis: Make sure that if you select something which is existing, and this, something, like, to give you an example, has RAG capabilities, has,
Pantelis Monogioudis: all these kind of other additional capabilities, you should not use these capabilities. You are not allowed to use these capabilities. We need to build
Pantelis Monogioudis: the second… the components of this specialized kind of knowledge on your own. So the second component is,
Pantelis Monogioudis: Also, you're not allowed to use any AI browsers.
Pantelis Monogioudis: you know, AI browsers are the latest
Pantelis Monogioudis: sort of, solution to, we don't have revenue in the AI space. And, so, the OMAT, OpenAI released a browser, I think, last week, some sort, and others. We're not allowed to use this browser.
Pantelis Monogioudis: Definitely you need an… a server to serve an LLM.
Pantelis Monogioudis: Okay, this is where most of the difficulty is, especially when it comes to hosting this LM server somewhere with your own computers. You know, some computers are fairly weak, but there is some kind of,
Pantelis Monogioudis: Cobbility to use, Carbility from OpenRouter, with some kind of a typically small expenditure from your side to route.
Pantelis Monogioudis: Even to a free LLM, the requests.
Pantelis Monogioudis: Okay, so technically, people are using Olama or LM Studio. Who's using Olama already?
Pantelis Monogioudis: I'm an LM Studio.
Pantelis Monogioudis: Some people, okay, locally in their kind of laptops, and you are selecting a model, which in this case is, 2.5.
Pantelis Monogioudis: Note.
Pantelis Monogioudis: The model is suggested because it gives you some reasonable trade-off between reasoning and execution.
Pantelis Monogioudis: And, so it could be, probably a very good model to, to…
Pantelis Monogioudis: For the project. But obviously, it has a… it comes in multiple sizes, right? And depending on your memory and the situation you have in your laptop, you may decide to use a smaller size version, right? So, there's no universal as which size it would fit you, it will depend on your personal situation.
Pantelis Monogioudis: Obviously, if you have a desktop computer at home, you will host the LLM model at your desktop computer, not in your laptop.
Pantelis Monogioudis: And you can point to that desktop computer from your laptop to go and submit all your requests and responses from that desktop computer. That's also my advice if you want to become someone
Pantelis Monogioudis: Professional in this kind of space, get a desktop machine which is on 24-7, and it costs much, much cheaper than a very, very expensive laptop.
Pantelis Monogioudis: Okay, keep your laptop thin, but have that as a computer that you can use from NYU, using.
Pantelis Monogioudis: A software content scheme.
Pantelis Monogioudis: Alright, so here's the third part, the graph rack, which is a version of what we call a river of metageneration.
Pantelis Monogioudis: So what is a reliable augmented generation? It's the sort of augmentation that we actually can do to the LLM with some specialized knowledge. What is this specialized knowledge in this kind of project is the course content. What course is this course?
Pantelis Monogioudis: Okay, so you have video recordings, you have, links, you have the website.
Pantelis Monogioudis: and you have links in that website that points to other websites, or PDF files, right? So you have this kind of frame.
Pantelis Monogioudis: Sort of media types. And, you're going to take these media types, and this is obviously a very general kind of blob diagram, and, you're going to pass them through some kind of, parsing.
Pantelis Monogioudis: and store them as raw data in a NoSQL database. That's what typically is done. What is the most famous NoSQL database that we are using?
Pantelis Monogioudis: MongoDB. MongoDB is a typical choice for that. And, you know, then you are going to, create,
Pantelis Monogioudis: The filterization pipeline. In other words, you're going to take some kind of, chunks,
Pantelis Monogioudis: And, you are going to use an embedding model, to, sort of create, metadata about this chunk, and also vectors that represent this chunk of text.
Pantelis Monogioudis: transcriptions, you know, all sorts of stuff that you can, of course, do. I'll come to what it is mandatory, what is not mandatory, optional, and how to work out this kind of…
Pantelis Monogioudis: project. So, these, vectors are stored in vector databases typically, right? And, there are plenty of vector databases around.
Pantelis Monogioudis: That, people are using, and, I'll come to which one, potentially the best part, please.
Pantelis Monogioudis: Yeah, I think, the best vector database, at least in my view, is Postgres itself. It has… it's a relational kind of database, but it has also vector capabilities. So, it's very common to actually have
Pantelis Monogioudis: multiple systems, some relational, some vector systems that are… need to be instantiated, and many people prefer these days to have just one thing, and store there both relational information as well as tables, as well as vectors.
Pantelis Monogioudis: So, obviously you have a problem, some question now, that itself is going to become a vector in this kind of D-dimensional space.
Pantelis Monogioudis: And, is going to…
Pantelis Monogioudis: going to… sorry, this regular input is going to also become a vector. The retrieval mechanism will work out. In other words, what could potentially be the nearby vectors in this kind of vector, space?
Pantelis Monogioudis: And, some responses are going to be generated that are going to, be, hitting this,
Pantelis Monogioudis: Is that, is that, so, it makes sense, you have, the,
Pantelis Monogioudis: prompt elements, okay? So this is basically, sometimes you have to, create assistant prompt as well, so I think probably the first kind of diagram is better to this.
Pantelis Monogioudis: So where's your prompt? Prompt is, over here, right? So from a vector kind of database, you got the retrieval mechanism, and then you have, the prompt.
Pantelis Monogioudis: that it is, hitting the LLM, and with these kind of additional learnings, the LLM is able to retrieve the, sort of, the response based on not only what it already knew, but also the new, information that you have, stored there in the vector database.
Pantelis Monogioudis: Now, this diagram here involves some kind of, additional kind of step called LLM fine-tuning. I don't believe that we need to do LLM fine-tuning in this exercise. LLM fine-tuning requires development of a dataset, which is called InstructDataset.
Pantelis Monogioudis: And, this instructor that said, basically, is, questions and answers that are considered to be the perfect answer, or the kind of a ground truth, right? And for this type of questions, if you… many people are using a much
Pantelis Monogioudis: more careful LLM to create this kind of ground truth these days.
Pantelis Monogioudis: And create this product. You actually can use to do fine-tuning of that, of whatever model you have selected.
Pantelis Monogioudis: And, that's basically,
Pantelis Monogioudis: That's basically the kind of process of RAG. Over here, however, this is basically what we call baseline rug. Baseline rug is
Pantelis Monogioudis: This system over there.
Pantelis Monogioudis: And, people have, these diagrams are actually coming Out of this, book.
Pantelis Monogioudis: But you all do have external.
Pantelis Monogioudis: book for, child.
Pantelis Monogioudis: However, do not view this implementation.
Pantelis Monogioudis: Because, obviously, the book is not building what we want. The book is building the baseline components.
Pantelis Monogioudis: That, return chunks.
Pantelis Monogioudis: From, from the… from the, sort of input, from the, from the visualization.
Pantelis Monogioudis: So what we need is called GraphRack, and involves the development of a knowledge graph.
Pantelis Monogioudis: So what is this kind of knowledge graph?
Pantelis Monogioudis: Think about the logistics graph as, sort of a graph with notes being a concept.
Pantelis Monogioudis: And with edges indicating something. And that something could be some dependency. You cannot understand this concept unless you understand A, B, C, D, other concepts, and these concepts are connected with edges to that thing that the student asked, let's say, to explain.
Pantelis Monogioudis: So the GraphRag, the Atlantic GraphRag implementations, but this is the code base that I advise you to start.
Pantelis Monogioudis: you know, implementation. So, this code base is a… I will call it an anonical codebase that allows you hacking very easily. That other codebase, the baseline codebase that I do not want you to touch, is almost like a production-level RAG system, okay? You can take this graph rack codebase, and at some point convert it, maybe after this course ends.
Pantelis Monogioudis: to something that you can deploy, and you can show some kind of, I would call it familiarity with engineering aspects of AI.
Pantelis Monogioudis: But that's not a requirement, right?
Pantelis Monogioudis: Also, LandChain offers implementations. If you're using LandChain or GraphRad, you're free to use that implementation as well.
Pantelis Monogioudis: The, as I said, the chunkiness in the graph rack is, is used to ask the LLM to extract entities and relationships from its chunk. We then define, based on that, the nodes and edges of the graph.
Pantelis Monogioudis: And, we are…
Pantelis Monogioudis: Going to, potentially get some summary of the so-called clusters of that kind of knowledge graph, because sometimes it's good to have some kind of hierarchy, and if a concept comes out of nowhere, to know in a very gigantic graph which sub-area of that graph you're able to focus your attention.
Pantelis Monogioudis: And I've waited a dime.
Pantelis Monogioudis: We are not returning chunks like the previous diagram, but we're returning subgraphs.
Pantelis Monogioudis: As I said, if you want to learn something about,
Pantelis Monogioudis: sort of, elbow, a lower bound. In machine learning, you need to know about genesis inequality, and so on and so on. So you have to have this subgraph that tells you that.
Pantelis Monogioudis: So the milestones are very typical, so you have to do a development environment, and you need to select your tools, and you need to do the development in the Docker environment.
Pantelis Monogioudis: And the rationale of that is that at some point, you will have multiple docket point genes.
Pantelis Monogioudis: Okay, so, ingestion milestone.
Pantelis Monogioudis: Yes, I'm giving you all this information about the website. Obviously, you know the website, you know the recordings, how to download these recordings, and how, potentially, you can transcribe these recordings if transcription is not already available, I believe it is.
Pantelis Monogioudis: And the… some information about the node is provided over here.
Pantelis Monogioudis: And some information about the edges is provided over here.
Pantelis Monogioudis: So, you… So the deliverable here is to…
Pantelis Monogioudis: present, if you like, the knowledge graph, obviously, as a structure, and showcase some of the connections that make sense. You want to showcase, for example, two, three connections.
Pantelis Monogioudis: The two, three concepts, and how the sub-concepts are leading to those.
Pantelis Monogioudis: You will then, obviously need to test that this thing is kind of working, so you will actually need to implement,
Pantelis Monogioudis: Interfedia query, and then the,
Pantelis Monogioudis: The system should respond not with an answer to that specific query, necessarily, but a learning path that leads to that answer, okay?
Pantelis Monogioudis: So that is not a typical situation that you have, today, even in ChatGPT. You… they give you the answer, but then they ask you, do you want to… you know, then you follow up with other questions and things like that, right? If you don't have something on the answer. This… the situation here is slightly different.
Pantelis Monogioudis: Definitely you need to include the preferences.
Pantelis Monogioudis: The references will include, website pages, links, in other words, pages in a PDF file that this specific answer, on this specific part of who that answer is based on, and… or, even, references in a
Pantelis Monogioudis: That video, sliced the time slices of the, of the video, but this specific thing was, sort of, video segments, in other words.
Pantelis Monogioudis: All right, so, generate the answers from the simpler concepts to more complicated concepts, and we will publish, questions that you can, sort of, showcase your results, right? We'll give you the questions, and then you can just showcase the results.
Pantelis Monogioudis: The way to proceed is a bit kind of important. Many people tend to focus their attention into something, and then they forget that, you know, the deadline is December 7th.
Pantelis Monogioudis: Right? And then they panic, and they kind of go quickly through some kind of hacks to deliver whatever, right? My suggestion to you, as always, is to do a 360-degree coverage of the whole system first, even if the system produces kind of garbage.
Pantelis Monogioudis: or some kind of, basic stuff is not really fully there. And then to tune as you… as the time goes by, okay? To… to improve it as the time goes by.
Pantelis Monogioudis: So, within the next two weeks, you should have a system like this up and running.
Pantelis Monogioudis: Okay, and then you can spend the remaining, kind of, 3 weeks on improving it.
Pantelis Monogioudis: So that's the… that's the approach in our service.
Pantelis Monogioudis: Any questions on the project?
Pantelis Monogioudis: Now, I know that you like to work alone, but also some people want to work with partners, right? So, you are free to choose a partner, one partner.
Pantelis Monogioudis: And, the partner, the partnership, just like in marriages, they have pros and cons, okay? So, one typical situation I've seen over and over again.
Pantelis Monogioudis: Is that, one partner is doing the work, and the other partner psychologically supports the one partner, or it's just observing.
Pantelis Monogioudis: ready to… ready to, sort of intervene, but that never happens. The worst thing you can do is to let this situation pass even by one week.
Pantelis Monogioudis: And then come to me after 4 weeks and ask, you know, I did everything myself, and this guy didn't do anything.
Pantelis Monogioudis: Okay? So, what is the best course of action? Leave the partnership, okay? The situation is not typically improving, even if you highlighted that, okay? So, leave the partnership, and when you do so, you need to inform us, you need to open a ticket.
Pantelis Monogioudis: And inform us.
Pantelis Monogioudis: It's about that. You left the partnership, and now you're going to develop the project on your own. So select carefully a partner that you are familiar with, or you have high hopes that this partnership will work. Otherwise, do it yourselves.
Pantelis Monogioudis: Okay, any questions?
Pantelis Monogioudis: So, as you can see, even the project is very easy. So now, I will go to the sec… start the second kind of part, with, by the way, when is your assignment, next assignment?
Pantelis Monogioudis: All right, November 2nd it is, I hope that you find it, kind of useful.
Pantelis Monogioudis: Alright, so let's start.
Pantelis Monogioudis: I hope you are recording your attendance.
Pantelis Monogioudis: And don't, tell me that you need to sign and have 20 people signing at the end of the lecture. Okay, so pass the paper or no. Alright, twice, if it needs to. Okay, so let's see. This is a lecture.
Pantelis Monogioudis: Something. Seven.
Pantelis Monogioudis: And the delay per day is, 24, 10.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: one month to the Thanksgiving break, I think.
Pantelis Monogioudis: I think. Oh. So…
Pantelis Monogioudis: Let's see how that goes. Okay, so, so today we're going to start, with some kind of the basic principles of NLP, and we'll obviously start with natural language policy, and,
Pantelis Monogioudis: We will be focusing primarily, although we'll cover some other additional kind of tasks, primarily on language modeling.
Pantelis Monogioudis: the discussion initially will be, going through the following kind of trajectory. Some basic principles to understand what kind of tasks we are going to be, sort of doing. We will be, talking about
Pantelis Monogioudis: Tokenization, which is a process via which the words are converted into numbers.
Pantelis Monogioudis: And we are going to start, on creating the bedrooms. So this is basically the plan, for the bedroom.
Pantelis Monogioudis: Obviously, if time allows, we'll continue on the first family of neural networks that are sort of able to process and create very simple, trivial, almost language models. So, although the RNNs are not really used today to
Pantelis Monogioudis: On large language models, they are a good instructive step to understand what transformers, right, and what comes after that.
Pantelis Monogioudis: So,
Pantelis Monogioudis: So we will then close the discussion by focusing then on transformers architectures, okay? What we think that we don't necessarily need to discuss here in this kind of class, in this kind of introductory class, is the engineering aspects of evidence, okay? So there are…
Pantelis Monogioudis: other courses, potentially, here at NYU, I do not know for sure, or better books that are covering just the engineering aspect of large language. So here, as we discussed earlier, we will be covering consonants.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: All right, so let's see. If you want some guidance onto this LLM engineering works, I can provide.
Pantelis Monogioudis: Okay, so, let's, see the natural language processing. So now we are going to be discussing a lot about text. Obviously, these days we have, capabilities to do
Pantelis Monogioudis: To bring text and computer vision together, or other modalities together, to understand, to ground better.
Pantelis Monogioudis: our responses from either of the modules provided, but in the kinds of pure text, the first thing that is going to be happening is that you're going to be taking a, let's say, a corpus of documents, a corpus of, actually, documents, and you are going to be,
Pantelis Monogioudis: Segmenting it into chunks.
Pantelis Monogioudis: And then you're going to tokenize it in the simplest form of tokenization, you're going to take its word and say that this is basically the token, okay? We'll see in the tokenization discussion why this is not probably the best idea.
Pantelis Monogioudis: And then, of course, there are a range of tasks, right, that are, going to provide some better clarity as to what is being said, or try to extract some form of a meaning.
Pantelis Monogioudis: So, part of speech stagging is one of them, and as you can see, the part of speech tagging results into some form of classification as to, okay, London is a noun.
Pantelis Monogioudis: this is a verb, and so on and so on. So you go and look at this kind of grammatical pattern, if you like, that emerges out of this kind of text.
Pantelis Monogioudis: Many times, this part of speech, kind of tagging is not necessarily an integral part of language modeling, potentially, but it can be used in some kind of context to allow better understanding of the meaning.
Pantelis Monogioudis: I'll skip the limitation, which is actually, trivial,
Pantelis Monogioudis: And, and I come to this kind of dependency parsing, which is, probably, probably, kind of, important, because, step, because in many instances, we want to identify, to take, if you like, some kind of text and create a tree out of it.
Pantelis Monogioudis: Where the root node of the tree needs to be identified, and…
Pantelis Monogioudis: words that are in the kind of, sort of down that kind of three, in the three kind of trajectories needs to be connected with other words. So, for example, in this, sentence, London is the capital, or the most populous
Pantelis Monogioudis: CD. Then you can actually connect
Pantelis Monogioudis: that is with, has a subject of London, and, the capital is an attribute, to London, effectively, right? So you can actually create these kind of relationships, that are able to, sort of help you, again, extract a meaning.
Pantelis Monogioudis: Ultimately, we have, some more,
Pantelis Monogioudis: additional kind of tasks, like name energy recognition. Name identity recognition is, a task that, let me see if this thing works. Yeah, so…
Pantelis Monogioudis: present?
Pantelis Monogioudis: Okay, yeah, it worked. So, you have, if you like, a text over here, and you want to go around the text and actually highlight things that are recognizable, to either a country, or a date, or an organization, or a name.
Pantelis Monogioudis: So over here, we have, this kind of, probability using,
Pantelis Monogioudis: I went on to NIP becomes Library Paul Stacy.
Pantelis Monogioudis: And this is, you know, certainly capability that is a single line of code in spacing, but obviously it requires some kind of algorithms to be run in the background to produce this result. And of course, nothing is perfect. For example, over here.
Pantelis Monogioudis: you, see that this is correctly identified, but this is not. This guy is a doctor, but, name and recognition highlighted to be, Maryland.
Pantelis Monogioudis: the state, okay? So, obviously, mistakes will actually happen. But in any case, going back, core reference resolution, which is the last task I want to kind of high value.
Pantelis Monogioudis: It's, a task where…
Pantelis Monogioudis: You know, you establish some, sometimes, some long, long in terms of, in the future, you need something like an int, and the computer must be able to associate it with long.
Pantelis Monogioudis: So this…
Pantelis Monogioudis: This means that, there will be, a need, if you like, whatever we develop here, to actually
Pantelis Monogioudis: You know, be able to accommodate, this kind of correspondences that will be spaced, quite, far away from each other.
Pantelis Monogioudis: This is zero.
Pantelis Monogioudis: This is quite a financial revolution.
Pantelis Monogioudis: Okay, so this kind of tasks are… right now, it's like a… almost,
Pantelis Monogioudis: sort of superficially kind of presented. So we'll go now into a little bit more technical details about tokenization.
Pantelis Monogioudis: Thank you.
Pantelis Monogioudis: So let's, let's look at tokenization.
Pantelis Monogioudis: Okay, text organization.
Pantelis Monogioudis: It's the right organization.
Pantelis Monogioudis: So…
Pantelis Monogioudis: So there are two, broad kind of categories in, kind of, tokenization, and the only thing that is different
Pantelis Monogioudis: is the resolution that they do. As we just discussed, there's this kind of world, reward level.
Pantelis Monogioudis: Where every word becomes, like, a token. And, there's a sort of a sound word.
Pantelis Monogioudis: Level, which is basically what most people are implementing.
Pantelis Monogioudis: Okay, so the tokenizer is going to take some kind of text at the input.
Pantelis Monogioudis: And it's gonna respond with some kind of indigo numbers.
Pantelis Monogioudis: And so on and so on. Each number will actually indicate an entry in the vocabulary that we try to build. So our job here is to build a vocabulary that we'll be calling capital V,
Pantelis Monogioudis: And in that vocabulary, all of these subwords, all words of whatever level we decide to do, is going to be listed as symbols, okay? Typical sizes of vocabularies could range from simple vocabularies, like 10,000,
Pantelis Monogioudis: entries, typically Alexa devices at your house, smart speakers, know about, around 10,000 kind of, words. If you ask them about thermodynamics, probably they will not answer, correctly. And, but, obviously, we are going to have, far more.
Pantelis Monogioudis: sizes, and actually much larger sizes, and the size of the vocabulary is going to play an important role in our complexity of the models that we're trying to build here.
Pantelis Monogioudis: So, okay, so, what is really the… so out of the many sub-work-level kind of tokenizers that exist.
Pantelis Monogioudis: We are going to, introduce here, the, sort of, byte-parent coding.
Pantelis Monogioudis: And this is, most large language models today are using that, or BPE.
Pantelis Monogioudis: So I'm going to, this backpair encoding is typically, like many other things that we have done, is not, invented today. It was invented back in the…
Pantelis Monogioudis: 70s and 80s, where, storage has become, was a major thing, right? And people were moving around with, things
Pantelis Monogioudis: like, called floppy disks and things of that nature to store documents and stuff like that. So compression was a very, very important component there. And so it was generated, created.
Pantelis Monogioudis: out of the compression type of research. So ultimately, we wanted to minimize the total number of bits that we can represent something. Okay, so I'll start with a very trivial example.
Pantelis Monogioudis: I will just do two examples here, just to understand the technique.
Pantelis Monogioudis: AKK.
Pantelis Monogioudis: BC.
Pantelis Monogioudis: A, A, A, B, A, C, that's, let's say that this is the very unlikely input text, which is a toy example here. So, just like what the
Pantelis Monogioudis: the encoder suggests, I'm going to select two consecutive, let's say, bytes. AA, for example, here. I see that we have this AA byte, sorry, sequence as a subsequence as a… in two places here, and I'm going to replace it with
Pantelis Monogioudis: another symbol. So, I will result… this will result into Z, A, B, D, Z, A, B, A, C.
Pantelis Monogioudis: And if I do exactly the same thing with capital Y now replacing AB,
Pantelis Monogioudis: This one will result into ZYD.
Pantelis Monogioudis: ZYAC.
Pantelis Monogioudis: And, finally, If I replace X…
Pantelis Monogioudis: to be ZY now. This will result into XDXAC. Okay.
Pantelis Monogioudis: So that's a kind of, sort of a trivial kind of substitution kind of, exercise.
Pantelis Monogioudis: obviously very deterministic, and what we have, what is the premise of pipeline encoding? The number of bytes that I need to represent these assignments, these three assignments, and the number of bytes that I need to represent this final string.
Pantelis Monogioudis: is going to be less than the number of bytes which I need to represent the original text.
Pantelis Monogioudis: Okay, that's the price.
Pantelis Monogioudis: So, a little bit more on the, kind of, engineering kind of side.
Pantelis Monogioudis: I will now use example number two.
Pantelis Monogioudis: which is also a famous, scene.
Pantelis Monogioudis: sales.
Pantelis Monogioudis: by… D.
Pantelis Monogioudis: sharp.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: So…
Pantelis Monogioudis: What we're actually doing, is, a bit more, you know, discussing a little bit how we are selecting
Pantelis Monogioudis: This kind of bites.
Pantelis Monogioudis: So, we are ranking.
Pantelis Monogioudis: every single character here, let's say, in terms of their frequency that is met in the input. So, for example, the space symbol here is met 5 times.
Pantelis Monogioudis: The S here is met 5 times. I may have done a mistake here, but I, you know, if you notice it, let me know. H3.
Pantelis Monogioudis: E… 5.
Pantelis Monogioudis: Adam?
Pantelis Monogioudis: Board?
Pantelis Monogioudis: Oh, one.
Pantelis Monogioudis: R1.
Pantelis Monogioudis: V1.
Pantelis Monogioudis: Why? Why?
Pantelis Monogioudis: Tier 1.
Pantelis Monogioudis: Okay, fair enough, up to this moment in time. I'm just looking at how many times this… every character is mentioned in the input text, and I'm writing down a number next to it.
Pantelis Monogioudis: Okay, so what I'm going to do is I'm going to be probably be focusing, on merging bytes that are in a kind of greedy way. Okay, so I'm going to suggest here that I am merging into a single capital Y, S, and H.
Pantelis Monogioudis: And obviously, I will have now another sentence.
Pantelis Monogioudis: yeet?
Pantelis Monogioudis: Underscore means a space symbol now. I've got this basically space.
Pantelis Monogioudis: Saus?
Pantelis Monogioudis: Y-E-L-S… by the… Sure.
Pantelis Monogioudis: Sorry, not sure about your…
Pantelis Monogioudis: Okay, so then, if I, if I go now to the second kind of step, I will just,
Pantelis Monogioudis: obviously, over here, what I have now is I'll continue to have this symbol.
Pantelis Monogioudis: Now my S…
Pantelis Monogioudis: Because it has been substituted by this Y symbol, right, 3 times, right? The S now is mentioned a smaller number of times, right, in the next conversation. Let's assume that this is the J2 I have here. Some other symbols that I had initially will have now a 0.
Pantelis Monogioudis: In front of them, right? I'm not sure if it is the H here, because there is one H. In fact, that was the mistake. But, I mean, this is a limited example. Some symbols will have a zero now, okay? And these symbols are going to be eliminated from the vocabulary.
Pantelis Monogioudis: In this specific step. So, I'm going to, however, in the original vocabulary, which I'm not going to mention, I am going to append
Pantelis Monogioudis: my new merge symbol, SH.
Pantelis Monogioudis: And quote, also, the number of times that this is, mentioned.
Pantelis Monogioudis: Then I'm now ready to start the next substitution. The next substitution is Z is YE.
Pantelis Monogioudis: Yeah, this, results into… Z underscore S-E-L-S, Z-L-L-S body… Vets.
Pantelis Monogioudis: Your?
Pantelis Monogioudis: And it's kind of a similar way, Some of these, letters or characters.
Pantelis Monogioudis: Are going to be eliminated by this substitution.
Pantelis Monogioudis: Some others will be reducing in terms of frequency.
Pantelis Monogioudis: Definitely what I would have at the end of the day.
Pantelis Monogioudis: I'm going to have…
Pantelis Monogioudis: all this kind of, merge kind of bytes at the end of this list, if I may call it later.
Pantelis Monogioudis: So what is going to happen is that, After a number of iterations, I will obtain
Pantelis Monogioudis: a vocabulary V, which is a set of this kind of, substance.
Pantelis Monogioudis: And the commonality of this set is going to be, as I said, to be the main driver in terms of our complexity. Some things will depend on this capital V from now on, okay? So, this is basically the vocabulary.
Pantelis Monogioudis: And this operator here is called cardinality. I'm gonna use it sometimes.
Pantelis Monogioudis: Of the set.
Pantelis Monogioudis: Sweet.
Pantelis Monogioudis: So we are going to build… yes, okay. So how do we choose the characters work?
Pantelis Monogioudis: Sorry? How do we choose which characters to, like, combine? Like, how do you choose Y for Twice it? How do you know? So, we are using it, we are using, this frequency indices, right? So it doesn't tell you which characters are, like, you know, followed by which other characters. This is just individual frequencies.
Pantelis Monogioudis: What do you mean that he doesn't tell a lot? I'm saying I selected SH to be, let's say, this guy, and, what is this,
Pantelis Monogioudis: this guy, right? Yeah, but that's only… so, but they could appear separately also. They don't have… they could appear separately, but I'm substituting only in the places where they are next to each other. That's called a bike appears.
Pantelis Monogioudis: 21 inches next to each other, unfortunately. Right, but, in that case, we don't know, look, just looking at the table, right, that they will follow each other or not.
Pantelis Monogioudis: But we start with a list, right? This is a table. Then we, if we make this substitution, right, which is, based, let's say, on this, you know, this guy's a high frequency kind of number.
Pantelis Monogioudis: And, we may actually select some other number next week, let's say. Let's say in this case it's 3, then this results into this, and then this results into that, and so on and so on. Right, so I'm building, if you like, a vocabulary based on this merger, incremental merchants.
Pantelis Monogioudis: Right, so the, say, suppose, like, we chose S to be our first character, so how do we know which character should be next? Is it the next entry in the table?
Pantelis Monogioudis: No, not necessarily. We may actually choose any other number, and see, you know, for example, we actually choose L, right? But obviously, we are, sort of selecting H here, because we result into some kind of a gritty way, to some kind of,
Pantelis Monogioudis: reduce kind of entropy in that list. So we, like, training random combinations like earlier? Yeah, one way of actually doing it, is to, you know, with having internally some kind of entropy, sort of, counter, right, and a measurement of the entropy of a set. And then we can choose it, again, creatively by.
Pantelis Monogioudis: Choosing something, and we will reduce that comes naturally.
Pantelis Monogioudis: Thank you, absolutely.
Pantelis Monogioudis: The maximum cost per reduction. It's almost like a decision tree kind of thing.
Pantelis Monogioudis: So, there are obviously many algorithms which are being used behind these kind of concepts, right? So, there are implementations in the kind of timing phase library, that are will… I don't think I have notes for those, but I'll…
Pantelis Monogioudis: Show you something, which is,
Pantelis Monogioudis: Yeah, over here I have some kind of a, nothing.
Pantelis Monogioudis: Yeah, I think, I think over here, I have something about The tokenizer, from,
Pantelis Monogioudis: from the transformation of library. So the specific organizer is being used there to, to do this kind of organization.
Pantelis Monogioudis: So there are a lot of details which I'm kind of bypassing here, because this discussion will actually be a little bit on the dry side, right? But however, tokenization is an important piece, because, as I said, it results into a vocabulary that is going to be quite important for us moving forward.
Pantelis Monogioudis: So you can take a look in these kind of implementations as well. There's a famous,
Pantelis Monogioudis: There's a famous web application called, TikTok9izer.
Pantelis Monogioudis: And I think an iframe here, I hope it is working. It allows you to select the tokenizer used in a kind of a language model.
Pantelis Monogioudis: And and type, if you like, a text.
Pantelis Monogioudis: In this case, we can type a text or a code.
Pantelis Monogioudis: fragment, or Python, let's equivalent here, and it will, tell you exactly how many tokens it results to. And that's kind of… I actually mentioned that, because it has a lot of importance with respect to…
Pantelis Monogioudis: why the tokenizers and the model, somehow they need to go hand in hand. Okay, so that's the point I'm trying to make. Okay, so let's copy a text here.
Pantelis Monogioudis: So… Let's copy this guy.
Pantelis Monogioudis: If I can't copy it. Come on, come on, copy it?
Pantelis Monogioudis: Oh, no.
Pantelis Monogioudis: Okay, I can copy it.
Pantelis Monogioudis: Oh.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: Copies.
Pantelis Monogioudis: Come on.
Pantelis Monogioudis: Thanks.
Pantelis Monogioudis: This is a… This is an organizer that GPT-2 adopted.
Pantelis Monogioudis: that this results into 186, tokens, right? Now…
Pantelis Monogioudis: If I select another kind of tokenizer.
Pantelis Monogioudis: If I select another tokenizer.
Pantelis Monogioudis: organizer.
Pantelis Monogioudis: which has, GPT-4, let's say GPT-4.
Pantelis Monogioudis: Sorry, something's changed here.
Pantelis Monogioudis: Something changed, and I don't know why it doesn't do that. This text was deleted.
Pantelis Monogioudis: Proposals.
Pantelis Monogioudis: Anyway, I don't want to drag you too long. If you select another tokenizer, a bit more modern tokenizer.
Pantelis Monogioudis: then the number of tokens that could result from that same piece of code is going to be a smaller number, right? This has some implication I want to mention. Obviously, as we just saw, the ability for the same kind of piece of code is
Pantelis Monogioudis: results into a larger number of tokens, what is going to change? It's going to change something which we'll all be calling context, the size of the context, right? So, the better the token, better in terms of,
Pantelis Monogioudis: the compression capability of the tokenizer is important because it will reduce, potentially, the context size. And this context size will actually be important when we discuss, the, sort of, both in RNNs, but also in transformers.
Pantelis Monogioudis: how they work. They would define certain things for us, how they work.
Pantelis Monogioudis: So there's some kind of discussion over here in terms of,
Pantelis Monogioudis: what kind of words, what kind of tokens there were present in GPT-2, so GPT-2 was very badly.
Pantelis Monogioudis: And the reason why it was very bad in coding, the tokenizer in coding was creating far more number of tokens compared to another next generation, where they introduced
Pantelis Monogioudis: coding as part of their, sort of tokenization strategy, and, they were able to, let's say, replace with one blocker, let's say, a Python keyword.
Pantelis Monogioudis: Great.
Pantelis Monogioudis: Are you following?
Pantelis Monogioudis: So the number of tokens that's coming out of this tokenized is actually important for various reasons, both memory, but also the ability to, sort of perform well, from the various kinds.
Pantelis Monogioudis: Input text that they have seen.
Pantelis Monogioudis: Okay, there's some discussion in the standard part of it.
Pantelis Monogioudis: Okay, along these lines. So what I'm going to do next, what is the next thing that happens after the organizer? The next thing that happens is the embedding. So what we want to do now, let's see, is to take these integral numbers.
Pantelis Monogioudis: Right? So here, in this vocabulary, the final kind of vocabulary, we have an integer number 1, 2, cardinal QV. I'm going to take each of these kind of numbers, and I will map them into a vector.
Pantelis Monogioudis: Okay, so I'm going to take this mapping, integral to vector, so we need to see how this will be done.
Pantelis Monogioudis: So these embeddings are, that we're going to do first, are called context-free embeddings, contextual-free embeddings.
Pantelis Monogioudis: They are, in other words.
Pantelis Monogioudis: Despite that, I'll be using the word context in my discussion. At the end of the day, we will need to persuade ourselves that they are contextual-free. In other words, they do not adjust themselves based on what other tokens are next to them, okay? So we'll… we'll do that.
Pantelis Monogioudis: So this is the first one, is this embedding.
Pantelis Monogioudis: Discussion?
Pantelis Monogioudis: context reactions.
Pantelis Monogioudis: When we go to Transformers, we'll see how contextual events will be created.
Pantelis Monogioudis: So what is this, encoder that we're gonna build?
Pantelis Monogioudis: This encoder is going to take some kind of sequence of tokens.
Pantelis Monogioudis: And it will map it into a vector.
Pantelis Monogioudis: These, vectors will be… part.
Pantelis Monogioudis: of, D-dimensional A set of real numbers.
Pantelis Monogioudis: The dimensional, Vector space.
Pantelis Monogioudis: And, it will,
Pantelis Monogioudis: So if you have, if you like a sequence, Of indicate numbers.
Pantelis Monogioudis: 1, 5, let's say 35.
Pantelis Monogioudis: The animals have an X1535 with a specific index.
Pantelis Monogioudis: that it is going to come out of this encoder, and this, the same thing will actually happen for all my, sort of, vectors. And then the question becomes, how do we…
Pantelis Monogioudis: create these vectors, based on what principle we're creating the vectors. Ultimately, we need to have some underlying, sort of objective, and not map vectors in a kind of a trivial way. What is the most trivial way you can think? You can take an integral number and convert it to a vector?
Pantelis Monogioudis: Thank you, everyone.
Pantelis Monogioudis: Why not equity?
Pantelis Monogioudis: one code encoder. I know that's the more trivial way you can think. It's a valid encoder.
Pantelis Monogioudis: And in this kind of case.
Pantelis Monogioudis: I have here… let's assume that I'm doing world-level organization, forget about biper quantum for this specific discussion, and I have now the mapping of the hotel. Well, the hotel is one entry in my
Pantelis Monogioudis: V, in my vocabulary, I go to the specific index where the hotel is mentioned, and I replace it with 1, and all the other numbers are
Pantelis Monogioudis: Zoo.
Pantelis Monogioudis: That is the one-hot encoding of hotel. And if someone also has, Mortel.
Pantelis Monogioudis: Also, and the motel is now a different entry in my vocabulary.
Pantelis Monogioudis: The, 001000, whatever.
Pantelis Monogioudis: Then this is the one called encoding of the motel, okay? Now, if I ask you.
Pantelis Monogioudis: The motel and the hotel are, entities that are conveying a similar meaning or not?
Pantelis Monogioudis: Yes, they are. They convey a very similar meaning. So…
Pantelis Monogioudis: What we were doing up to this moment in time to… what operation we're doing over and over again in this kind of course.
Pantelis Monogioudis: If there's one operation we're doing over and over again in this course, what was that?
Pantelis Monogioudis: dot?
Pantelis Monogioudis: product, right? Because we were trying to, sort of…
Pantelis Monogioudis: have some kind of indication on how my W vector should be, adjusting this X vector, so we have W's as projects all the time. So if I do now…
Pantelis Monogioudis: Hotel, transpose motel.
Pantelis Monogioudis: what I'm getting.
Pantelis Monogioudis: I'm getting S is big fat zero. This means that
Pantelis Monogioudis: In this vector space, in this specific encoder, the…
Pantelis Monogioudis: two vectors that I'm coming up with of, you know, zero dot product. That's not a very desirable thing. What I want to see is I want to see this. I want to see a construction where vectors that correspond to a world of similar meaning are
Pantelis Monogioudis: close to each other. So, what I'm gonna do… is,
Pantelis Monogioudis: I'm going to ultimately construct this.
Pantelis Monogioudis: All right, so I have right here a vocabulary of 10,000 entries. Each point of the vocabulary is, one entry, each point represents one entry in my vocabulary.
Pantelis Monogioudis: And, obviously, this is, I'm going to be building, a dimensional vector, a vector space of dimensions 200. Let's say my small letter D is now 200. Okay? How I arrived in this kind of 200? It's kind of an interesting question.
Pantelis Monogioudis: Well, typically.
Pantelis Monogioudis: This has some information theoretic kind of basis, which is kind of quite interesting. But typically, the larger the vocabulary is.
Pantelis Monogioudis: the larger the D should be.
Pantelis Monogioudis: Okay, it has something to do with, sort of capacity, okay, information.
Pantelis Monogioudis: But, here, this offer selected 200.
Pantelis Monogioudis: Obviously, we cannot visualize 200 dimensions. Therefore, we apply
Pantelis Monogioudis: the well-known principle component analysis to reduce this number of dimensions down to 3. This is the point cloud that we are seeing in three-dimensional tracking.
Pantelis Monogioudis: However.
Pantelis Monogioudis: What you see here on the results of these assignments from integer numbers to vectors, you will see what happens in a 200-dimensional space, okay? So if I type a word
Pantelis Monogioudis: Car.
Pantelis Monogioudis: Right Protect the work card.
Pantelis Monogioudis: And select it, you have to select it.
Pantelis Monogioudis: Come on.
Pantelis Monogioudis: You're about to select it, so…
Pantelis Monogioudis: These are the nearest points in the original space. Original means 200-dimensional space. Next to the whole car, I'm seeing driver, I'm seeing cars, automobile, race, racing, vehicle, and so on.
Pantelis Monogioudis: So this is basically what I want. I want to achieve that probability, and now the next discussion is…
Pantelis Monogioudis: how I'm going to do that.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: So… Many years ago, and surprisingly, an intern.
Pantelis Monogioudis: In a company, I think it was Microsoft.
Pantelis Monogioudis: came up with this kind of idea, okay? So, apparently, if you are interning in Microsoft, they don't make you
Pantelis Monogioudis: to the firm.
Pantelis Monogioudis: make coffees, but, they actually ask you to do something of value. So they, so this,
Pantelis Monogioudis: This beddings, was… took the name Wartubecker Bedding. Okay, so I'm gonna be discussing…
Pantelis Monogioudis: Now they're working back and order.
Pantelis Monogioudis: That will achieve this, or we'll achieve this, the,
Pantelis Monogioudis: This is a point cloud behavior, and in the point cloud we've seen, this is, in fact, a water vacant problem.
Pantelis Monogioudis: So, what, what Juvec encoder will actually do?
Pantelis Monogioudis: is, we'll, take a word.
Pantelis Monogioudis: And we'll encode it to achieve that, and becoming this kind of idea, there was some kind of, you know, many, many years before, you know, the invention of Word2Vec, there were many linguists, right? They were creating some kind of a classical
Pantelis Monogioudis: language models, like the bigram, the trigram, and things like that. They were all… all of them were frequency kind of oriented, so they… they were, like, pretty primitive. But one guy and one linguist.
Pantelis Monogioudis: Okay, so let me go back to the site.
Pantelis Monogioudis: If I can't.
Pantelis Monogioudis: There were one guy called, Ferd.
Pantelis Monogioudis: suggested the following. This is basically the inspiration behind this kind of construction.
Pantelis Monogioudis: This idea was formerly called semantics. And basically, it's a very simple kind of concept. So, he said that for an awards meeting.
Pantelis Monogioudis: It's given by words that frequently appear close by.
Pantelis Monogioudis: So, for example.
Pantelis Monogioudis: If you are, in the world of, finance, right, if your corpus of documents or credit documents are associated with finance, behind… next to the World Bank, probably you will see words like money, you will see institution, you will see building, or whatever. Something like that, right?
Pantelis Monogioudis: Now, if the corporation comes from National Geographic.
Pantelis Monogioudis: Next to the World Bank, you will see what?
Pantelis Monogioudis: carpodice, or something. Something associated with that, right? So, so…
Pantelis Monogioudis: So the idea is the following.
Pantelis Monogioudis: I'm going to be… Looking what is going on around me.
Pantelis Monogioudis: Right?
Pantelis Monogioudis: To effectively use them, them being the world around me, to map myself in this D-dimensional space, in the right place, in their neighborhood.
Pantelis Monogioudis: That is the core idea. Okay, so we'll see now how this can be mechanized, in this kind of construction.
Pantelis Monogioudis: Okay, let's see.
Pantelis Monogioudis: So, the encoder, Well, the encoder will produce
Pantelis Monogioudis: for a window size that, unfortunately, is called a context, and that's why many people get confused that this is a contextual encoding. It's not.
Pantelis Monogioudis: But I'll call it window. Let's say two words behind the center word, and two words after the center word. And pictorially, you can actually see them right here.
Pantelis Monogioudis: I will convert the mapping problem from an integral number to a point in this D-dimensional space into a prediction problem.
Pantelis Monogioudis: There's also the reverse prediction problem as well, but right now it's going to be the so-called forward prediction problem, where
Pantelis Monogioudis: This word… Being into, being into.
Pantelis Monogioudis: can predict
Pantelis Monogioudis: This center will therefore take the specific kind of index, can predict banking, and can predict crisis, and can predict learning, and can predict problems.
Pantelis Monogioudis: So these are all… conditional probabilities, right? So…
Pantelis Monogioudis: We have done a lot about conditional probabilities.
Pantelis Monogioudis: Obviously, we're… all of this time, we're labeling how to do destination of B of what?
Pantelis Monogioudis: Be off.
Pantelis Monogioudis: P of Y given X. That's what we've been doing all along, right? So we were estimating a conditional probability distribution.
Pantelis Monogioudis: So over here, we will be estimating
Pantelis Monogioudis: also a conditional probability distribution, because based on that, we will be able to predict these words that are around. So, obviously, after a while, after the next word, this world becomes the banking, and some other words are next week.
Pantelis Monogioudis: And obviously, we have…
Pantelis Monogioudis: I will call it a training kind of data set that we can develop purely based on our text.
Pantelis Monogioudis: So, I need labels, so here is my way that I would construct trivially the labels. You have an input text called, Claude Monet, I painted the Grand Canal of Venice in 1908.
Pantelis Monogioudis: every single…
Pantelis Monogioudis: word in this kind of text, has two words behind it, two words after it, so I'm going and creating this kind of training data set.
Pantelis Monogioudis: So, I have now ground truth.
Pantelis Monogioudis: that I can start building something, that it will
Pantelis Monogioudis: most of the time, right, resemble the probabilities of my, of my training kind of data set. In other words, it will go into… since this is my
Pantelis Monogioudis: This is my underlying P data distribution, right? There's a P data here, right? My P model, similar to what we have done always, should be something very close to P data hat. So, what we will do…
Pantelis Monogioudis: I will denote, in other words, with a convenient notation, Y hat, the output of this encoder.
Pantelis Monogioudis: of P minus 2, Y hat of t minus 1, Y hat of P plus 1, and Y hat of P plus 2. Note, of course, I don't have Y hat of P.
Pantelis Monogioudis: Because I am… Using the word T. The T word is the central word as input.
Pantelis Monogioudis: And this is what is basically going to be called the window.
Pantelis Monogioudis: A window of 2 means plus or minus 2.
Pantelis Monogioudis: if I want to take them.
Pantelis Monogioudis: Claude Monet sentence, for example, and convert it to some, sort of a bit more formal kind of thing, right? In fact, I take all these kind of sentences. I definitely have present in my
Pantelis Monogioudis: documents, a P data hat, which will have the following kind of format.
Pantelis Monogioudis: W, T minor C, WT minus C plus 1, comma, dot dot dot, WT plus C.
Pantelis Monogioudis: Do you agree?
Pantelis Monogioudis: Given… WT.
Pantelis Monogioudis: This is my central world.
Pantelis Monogioudis: And this is gonna be the… context.
Pantelis Monogioudis: Window, in other words.
Pantelis Monogioudis: worlds.
Pantelis Monogioudis: So what I'm going to do now is I'm going to create a P model, right? A P model, like, always I was doing that. I was going to create a probabilistic model.
Pantelis Monogioudis: That, well, surprisingly, this probabilistic model should have exactly the same
Pantelis Monogioudis: sort of, expressing exactly the same conditional probability distribution like the P data hat, that's what I want always, so it will be something like, WT minus 3.
Pantelis Monogioudis: Comma, dot dot dot, WT plus C.
Pantelis Monogioudis: given, WT, comma, T-Dial.
Pantelis Monogioudis: It's obviously a parameterized conditional probability distribution. Remember, always the P model of,
Pantelis Monogioudis: Why give an X comma W? That's exactly the same thing.
Pantelis Monogioudis: Now, obviously, this is a conditional probability distribution, right? This…
Pantelis Monogioudis: traditional probability distribution, that it is, quite, gigantic, in a sense that, you know, let's assume I have, each one of these kind of W's can take any of the 100,000 possibilities in my vocabulary. So I have a number of
Pantelis Monogioudis: 100,000 dimensional kind of things before the given kind of symbol, and I have the… another 100,000 thing after the given symbol. So what I'm trying to do here is I want to find a way, similar way as we have done also in the maximum likelihood kind of discussion, to simplify the probability distribution.
Pantelis Monogioudis: So, I'm going to adopt,
Pantelis Monogioudis: a property, I mean, a rule that, is being used, sort of, frequently for this.
Pantelis Monogioudis: It's called Knife Base.
Pantelis Monogioudis: assumption.
Pantelis Monogioudis: Okay, because…
Pantelis Monogioudis: It's, it's called naive because it is, I will call it, it's a strong assumption, okay?
Pantelis Monogioudis: And what is really the assumption of the naive-based assumption? Let me just open a parentheses here.
Pantelis Monogioudis: knife base.
Pantelis Monogioudis: So, is the following.
Pantelis Monogioudis: given… the center warm.
Pantelis Monogioudis: All context words.
Pantelis Monogioudis: are independent of each other. R?
Pantelis Monogioudis: of each other.
Pantelis Monogioudis: We did a very similar assumption, it was not a conditional assumption there, when we're discussing the
Pantelis Monogioudis: maximum likelihood. Basically, the assumption back then was that each sample that came out of this probability distribution is independently sampled from the other samples of the probability distribution, or the samples of the real line. So here we have conditioning, and that's why
Pantelis Monogioudis: We are going to express it this way, so let's assume, that this conditioning assumes the following.
Pantelis Monogioudis: is, results in the following. P model.
Pantelis Monogioudis: of WT minus C, given WT comma theta.
Pantelis Monogioudis: Times.
Pantelis Monogioudis: P-model.
Pantelis Monogioudis: of WT minus C plus 1, given WT comma theta.
Pantelis Monogioudis: times dot dot dot.
Pantelis Monogioudis: And, I'm going to express this as the product.
Pantelis Monogioudis: from J is equal to minus C.
Pantelis Monogioudis: to J different, plus C when J is different than 0.
Pantelis Monogioudis: of a P model.
Pantelis Monogioudis: of WT plus J.
Pantelis Monogioudis: Given WT comma theta.
Pantelis Monogioudis: I hope you're following the construction. It's basically a kind of a first principles kind of construction. If, of course, we believe that
Pantelis Monogioudis: that, this mapping is, of, of value, right?
Pantelis Monogioudis: So how are we… what are we going to do next? You have a P model now, and, which, optimization
Pantelis Monogioudis: Patina, you will be using to,
Pantelis Monogioudis: Come up with a theta star.
Pantelis Monogioudis: Which one?
Pantelis Monogioudis: Don't be afraid. Maximum like you. Maximum like you.
Pantelis Monogioudis: So, al?
Pantelis Monogioudis: my lost… This is the matching likelihood formula.
Pantelis Monogioudis: Now…
Pantelis Monogioudis: In the maximum lagging formula, we had X come up with expectation is, for all my examples that someone gave me, that are distributed according to the B data distribution. Now, what are my examples here? What is X? Before, it was X and Y. Which one is now X, and which one is the Y?
Pantelis Monogioudis: X is what? The center world. X is the center word, and Y now is there's a set of nearby words, right, a plus 2 and minus 2. Okay, so here we have…
Pantelis Monogioudis: I'm writing in… In the notation.
Pantelis Monogioudis: That they should be familiar with.
Pantelis Monogioudis: WT.
Pantelis Monogioudis: Blush… Jane?
Pantelis Monogioudis: According to the feed data, Card distribution.
Pantelis Monogioudis: log.
Pantelis Monogioudis: Of what?
Pantelis Monogioudis: You should remember this formula. Log off.
Pantelis Monogioudis: We did a hat, no, log of the P model. P model, that is. What is my P model?
Pantelis Monogioudis: This guy over here.
Pantelis Monogioudis: Let me write it, okay.
Pantelis Monogioudis: Wrong. It's not a promoter.
Pantelis Monogioudis: It's a product. Law of the product.
Pantelis Monogioudis: From J is equal to minus C to J is equal to plus C of P model.
Pantelis Monogioudis: WT plus J.
Pantelis Monogioudis: Given WT Pharmacy.
Pantelis Monogioudis: Are we following?
Pantelis Monogioudis: That is basically from the maximum likelihood
Pantelis Monogioudis: That we have, we have looked before.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: So we want to produce now, and…
Pantelis Monogioudis: to implement this kind of, to implement a hypothesis, right? Obviously, we understand the action-like criterion, but where is the P model?
Pantelis Monogioudis: Okay, so the P model…
Pantelis Monogioudis: will be produced similarly, right, out… using a neural network construction. I mean, there will be a neural network just like we have done in the past, but the neural network was doing, back then classification, prediction based on classification, right? Very similar things, though. It will be some prediction that will predict the nearby worlds, right? So, this network, however, will look a little bit different. While many of the networks
Pantelis Monogioudis: We have done in the past.
Pantelis Monogioudis: We're taking a very highly dimensional kind of input, and we're mapping it into a low-dimensional thing, right? A bedding, effectively, a bedding in some kind of D-dimensional space, and then we're using this D-dimensional representation
Pantelis Monogioudis: With the appropriate care.
Pantelis Monogioudis: To do the prediction.
Pantelis Monogioudis: Over here.
Pantelis Monogioudis: we are going to have this, obviously, but it will look a little bit different. So, we are going to have definitely,
Pantelis Monogioudis: I'm going to draw it now left to right. I'm going to definitely going to have a highly, sort of, dimensional kind of input. This highly dimensional input is WT. Why I call it now WT can put a vector there, because
Pantelis Monogioudis: Evidently, for the neural network, the world needs a vector at the input. This vector will be the trivial vector, the one called
Pantelis Monogioudis: Encoded vector of the word car.
Pantelis Monogioudis: So, this is one of me.
Pantelis Monogioudis: re-dimensional. This thing here is…
Pantelis Monogioudis: B-dimensional.
Pantelis Monogioudis: And, I'm gonna have some… Layer over here.
Pantelis Monogioudis: That is going to employ the dense matrix, W,
Pantelis Monogioudis: In a way that I won't… to construct,
Pantelis Monogioudis: representation, Z. This representation will be
Pantelis Monogioudis: belongs to the Art of the Power of Deep.
Pantelis Monogioudis: That's my Z vector here.
Pantelis Monogioudis: But this Z vector should be such.
Pantelis Monogioudis: This Z vector should be such that when it is fed.
Pantelis Monogioudis: remember the head that we always… we were doing, right? The head was a projection matrix, typically, down to the number of dimensions that we needed for classification, right?
Pantelis Monogioudis: Here, The, the predictor will be… Ahead that it will be…
Pantelis Monogioudis: predicting words in the original space, right? When Z is coming in, right, as a representation, the word that should come out will actually be
Pantelis Monogioudis: the JF word.
Pantelis Monogioudis: Of my vocabulary.
Pantelis Monogioudis: So ideally, what I wanted to see, what I showed you with this projector kind of demo,
Pantelis Monogioudis: Definitely, I want to have something which is… under V-dimensional, so a space.
Pantelis Monogioudis: And this…
Pantelis Monogioudis: This white hat, after training is done, you remember the first word that came out, the closest word that came out was Hulabrim? It was…
Pantelis Monogioudis: The result of taking.
Pantelis Monogioudis: producing.
Pantelis Monogioudis: a posterior probability distribution, right? Always the Y hat was a posterior probability distribution. How many elements I have in this posterior, though?
Pantelis Monogioudis: the elements. That's 100,000 entries, right? Right there. So one of these, the word that won the competition is going to be the word that corresponds
Pantelis Monogioudis: True?
Pantelis Monogioudis: driver.
Pantelis Monogioudis: So, dimensioning-wise, I need, here at V times D matrix.
Pantelis Monogioudis: Because I will form a Z to be WT transpose.
Pantelis Monogioudis: W.
Pantelis Monogioudis: I won't…
Pantelis Monogioudis: Simple of use.
Pantelis Monogioudis: Z, J prime, Prime, it doesn't mean I deliver it. Prime here means different.
Pantelis Monogioudis: that it will take the Z, and with the help of another matrix.
Pantelis Monogioudis: it will lift it up into the V-dimensional space. So the W matrix is doing the embedding, the projection, and the W prime matrix is doing the lifting.
Pantelis Monogioudis: back to the V-dimensional space.
Pantelis Monogioudis: And,
Pantelis Monogioudis: what else am I missing here? I forgot to mention, you know, here the ZJ prime is my logics, right? Again, 100,000 dimensions, right there, a vector.
Pantelis Monogioudis: But obviously, I need to feed this vector to…
Pantelis Monogioudis: a softmax, which I forgot to draw here.
Pantelis Monogioudis: Let me just draw the soft marks.
Pantelis Monogioudis: Okay, and this soft marks, obviously, We result?
Pantelis Monogioudis: Into the…
Pantelis Monogioudis: 3-dimensional posterior.
Pantelis Monogioudis: given the center word, what is the probabilities of all possible words I can have Next to me.
Pantelis Monogioudis: And after… initially, the words will be garbage, right? But after a while, after training is done, right, the lifting will result into… into what? What is going to be changing here during the training process?
Pantelis Monogioudis: What is going to be changing constantly during the training process?
Pantelis Monogioudis: What are the trainable parameters? These guys, obviously, right? These are what the work will change. This market will change, right?
Pantelis Monogioudis: Such that the Z vector moved from some input
Pantelis Monogioudis: Some position, right, a random position in the space, not into, but a random position in a D-dimensional space, moved
Pantelis Monogioudis: to acquisition, where… The other entries…
Pantelis Monogioudis: Driver, vehicle, and so on will be next.
Pantelis Monogioudis: And this is going to be happening, for all…
Pantelis Monogioudis: Words of my vocabulary at the same time.
Pantelis Monogioudis: Because in the next time, another center board will show up, which will have potentially a different meaning.
Pantelis Monogioudis: Right? Could be a part of a stop word, for example, a finished sentence, right? And then another sentence starts, which is… has a different meaning.
Pantelis Monogioudis: So… Students paid off their loans, full stop.
Pantelis Monogioudis: The university, blah blah, and so on. University and loans… Maybe they have similar meanings.
Pantelis Monogioudis: But, not necessarily.
Pantelis Monogioudis: Are you able to follow what's going on here? No? Okay. All right, let me explain this again. So, we… The analogy is the following.
Pantelis Monogioudis: We take something in the… 30-dimensional space, Let's say, like a ball.
Pantelis Monogioudis: And we will throw it in a lower dimensional space, the floor, In a specific…
Pantelis Monogioudis: place in this two-dimensional kind of law, right?
Pantelis Monogioudis: That when it is lifting itself kind of up, right?
Pantelis Monogioudis: Then we will have, a point in the three-dimensional kind of space that is,
Pantelis Monogioudis: That exactly the same,
Pantelis Monogioudis: location will be, but from a different point in the three-dimensional space, will be also finding in that similar kind of neighborhood, right? So, similar things are actually happening here. I have…
Pantelis Monogioudis: a word in the three-dimensional space, I'm embedding it, projecting it in the Z, in the vector Z, in a D-dimensional space, and this vector is such… this is basically the required embedding. This is basically what I'm after. I'm not after here, I'm not after this guy, I'm not after this guy, I'm just after this, the values of this vector Z. The value of this vector Z will be such that
Pantelis Monogioudis: When the specific matrix does the lifting, It will result into a…
Pantelis Monogioudis: A vector here whose largest element will actually be a probability that corresponds to the wartime.
Pantelis Monogioudis: Okay, that's, that's basically what it, what it is.
Pantelis Monogioudis: So there is a…
Pantelis Monogioudis: some kind of analysis, sometimes they're successful, sometimes not very successful, right? But, I did my best to try to
Pantelis Monogioudis: I tried to explain it.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: Nope.
Pantelis Monogioudis: So, what is going to happen now is that we… let me finish the forward equations. Y… sorry, the Y…
Pantelis Monogioudis: hat.
Pantelis Monogioudis: J is Shockmax.
Pantelis Monogioudis: of ZJ prime.
Pantelis Monogioudis: My loss is obviously the cross entropy.
Pantelis Monogioudis: Between what?
Pantelis Monogioudis: Y hat J.
Pantelis Monogioudis: and YJ. Do I know YJ? Yes, I do, because I have a training business set.
Pantelis Monogioudis: And obviously, I have a theta.
Pantelis Monogioudis: that it is… W, comma, W prime j.
Pantelis Monogioudis: But this is a situation, this diagram, is predicting how many nearby worlds.
Pantelis Monogioudis: Just one. The Jth one, right? What should happen, right, we are tasked to come up with a Z that will predict
Pantelis Monogioudis: All… for nearby wards, right? So… To… to do that.
Pantelis Monogioudis: In other words, I would blot the situation with just another one, let's say before and after.
Pantelis Monogioudis: Nope.
Pantelis Monogioudis: There is… what I'm saying is that
Pantelis Monogioudis: I can take the Z and adjust this WJ minus to produce the nearby war, but at the same time, I have to produce another nearby war that I'm, I'm,
Pantelis Monogioudis: I'm seeing frequently in my training data set. And then another and another. There will be four of these branches, in other words, right, in the network. And the training, just like what you have seen, kind of, in the past.
Pantelis Monogioudis: is going to do jointly optimizing the loss. You know, what is the loss? The cross-entropy, in this case, will be some kind of an additive
Pantelis Monogioudis: loss that will sum the loss associated with prediction with J, then the loss associated with prediction with J plus 1, the J-1, and so on and so forth.
Pantelis Monogioudis: So all of these kind of mathematics are jointly adapted by the… so basically, essentially the joint optimization that's actually going on to adjust them, such as the Z is such that it is already, as I said, B, in the neighborhood of these other four grains.
Pantelis Monogioudis: Finally, at the end of the day, We are going to have
Pantelis Monogioudis: Out of the scan of two mattresses.
Pantelis Monogioudis: Let's assume the first schedule, just one more. Out of the span of two matrices, which matrix do we need?
Pantelis Monogioudis: Don't do… they're dugging him.
Pantelis Monogioudis: if I had a W, right, I'll take the W, I will freeze it, right, at the end of the kind of training process, and I'll send it up to the hub.
Pantelis Monogioudis: To a hub, where people are able to download it.
Pantelis Monogioudis: And use it to create these embeddings for that specific vocabulary.
Pantelis Monogioudis: Could be an English vocabulary of science.
Pantelis Monogioudis: 10,000. Another vocabulary will be for French size 10,000, and so on and so on, right? There will be plenty of vocabularies. But definitely I will need the W star matrix.
Pantelis Monogioudis: that it will be… what is the dimension of the W star matrix?
Pantelis Monogioudis: will be… definitely we have zeros.
Pantelis Monogioudis: We use this dimension to match it in the forward pass, if you like. There will be V rows.
Pantelis Monogioudis: And, Dick Collins.
Pantelis Monogioudis: However, in order for me to produce the Z according to this kind of equation, I need to do.
Pantelis Monogioudis: Wt transpose W star.
Pantelis Monogioudis: For this specific theme, which is the…
Pantelis Monogioudis: Specific.
Pantelis Monogioudis: index, if you like. Remember, however, that this WT is one quad input.
Pantelis Monogioudis: So if I do the multiplication.
Pantelis Monogioudis: I don't need to do one locations with zeros, right? I only have one location with WT,
Pantelis Monogioudis: It's… I can just as well.
Pantelis Monogioudis: Pick up.
Pantelis Monogioudis: The specific row that corresponds to the index of the word that I am.
Pantelis Monogioudis: I want to calculate the bedding. So, if the work, as I did in the beginning, is
Pantelis Monogioudis: The word, let's say, bank, that whose index is, was indexing that.
Pantelis Monogioudis: in a tokenization kind of step was, you know, 359. This is the 359 row of this…
Pantelis Monogioudis: W-star matrix.
Pantelis Monogioudis: Pick it up, and I don't do any multiplication, and this will be my Z for the work function.
Pantelis Monogioudis: Now, this is the critical part.
Pantelis Monogioudis: despite the fact that I have used nearby words to build a prediction problem.
Pantelis Monogioudis: At the end of the day.
Pantelis Monogioudis: If I have a mixture of finance, news, and National Geographic News.
Pantelis Monogioudis: in my corpus.
Pantelis Monogioudis: Do I have multiple banks, or one bank?
Pantelis Monogioudis: I won.
Pantelis Monogioudis: This means that
Pantelis Monogioudis: I don't really see the context of this kind of world in this kind of a grand scale of things, because remember, we are seeing a representation which is the average, right, across the whole training kind of deals.
Pantelis Monogioudis: So if one word carries two different meanings, let's say the institution or the riverbank, Zed.
Pantelis Monogioudis: Simplistically thinking, the… The point in the D-dimensional space where this work will be mapped.
Pantelis Monogioudis: Is somewhere in the middle between the two.
Pantelis Monogioudis: Somewhere different, definitely, between either of the two.
Pantelis Monogioudis: So that's why… the reason why we have that, there's only… let me write it down.
Pantelis Monogioudis: There is only…
Pantelis Monogioudis: one vector.
Pantelis Monogioudis: poor bank.
Pantelis Monogioudis: irrespective.
Pantelis Monogioudis: Oh.
Pantelis Monogioudis: V.
Pantelis Monogioudis: Meaning?
Pantelis Monogioudis: that may be… with me.
Pantelis Monogioudis: B.
Pantelis Monogioudis: Sort of, contained.
Pantelis Monogioudis: Ian.
Pantelis Monogioudis: the, to go for input.
Pantelis Monogioudis: Lightning.
Pantelis Monogioudis: data.
Pantelis Monogioudis: So, in other words, what I'm saying is that in this kind of a D-dimensional kind of space, this point over here, if I,
Pantelis Monogioudis: If I map it, it will be associated with a bank of an institution, of the institution, with the meaning of institution.
Pantelis Monogioudis: If the training data were all finance news.
Pantelis Monogioudis: There'll be that point there.
Pantelis Monogioudis: There will be a different point, If it is National Geographic.
Pantelis Monogioudis: Different point.
Pantelis Monogioudis: Because the words next to it are different.
Pantelis Monogioudis: And then, so…
Pantelis Monogioudis: if a word exists in the vector B, does she only have one point in the, in the, graph. Yes. Yeah, so, so this will be there
Pantelis Monogioudis: Bunk?
Pantelis Monogioudis: associated with that either, right? With geography, right?
Pantelis Monogioudis: So, if I feed, however, into the construction both of these training kind of data, separate kind of domains, right, there will be something
Pantelis Monogioudis: That will be generated.
Pantelis Monogioudis: There is going to be some average between the two. I do not know exactly how it will end up. Also, there are some imbalances in the situation between the two different kind of data sets, but there will be a world bank.
Pantelis Monogioudis: For the mixture.
Pantelis Monogioudis: For the mix of the two.
Pantelis Monogioudis: Found, finally, Is that, this construction
Pantelis Monogioudis: This construction also had some, kind of nicer properties associated with analogies, right? So the… they were able to, to show empirically that,
Pantelis Monogioudis: Establishing an analogy between two things, right, is converted to arithmetic operation.
Pantelis Monogioudis: So, analogies, by the way, is not,
Pantelis Monogioudis: you know, many people suggest that analogy is also one of the key ingredients of intelligence. It could be the ability to create this kind of analogies between the two parts. For example, in this specific trivial example, you can see, you know, what a man is to a woman is what a means to the queen. These two distances are…
Pantelis Monogioudis: Almost identical to each other, and this is basically what allows us to establish mathematically a machine to be able to establish kind of finality.
Pantelis Monogioudis: That's the last point I want to mention about the construction of the World Quebec.
Pantelis Monogioudis: Obviously, in your side, you have, You, you have some…
Pantelis Monogioudis: From scratch implementation of War Quebec.
Pantelis Monogioudis: Don't have the… sentencing the purpose of the document to have,
Pantelis Monogioudis: a for loop that goes through one word in each kind of sentence. We have, go ahead.
Pantelis Monogioudis: Index of the sender world.
Pantelis Monogioudis: And, the conversion of the central world, one code and modding, over here.
Pantelis Monogioudis: You have a context window, and, your J is the index of the kind of context window.
Pantelis Monogioudis: And you build a training data based on that context window around. That's basically the…
Pantelis Monogioudis: train… how you develop the training kind of data, exactly as we discussed. Class and kind of helper functions here.
Pantelis Monogioudis: And then you have the forward pass, which I have discussed.
Pantelis Monogioudis: instead of Z, you have an H here, and we have the masses W and W prime, and you are doing, effectively, the Y-hat C.
Pantelis Monogioudis: Of the center word… of the… sorry, the Y-hat of the… that corresponds to the predictive worn out of the soft marks.
Pantelis Monogioudis: Which plays at the output of the W2 matrix here.
Pantelis Monogioudis: And then you have the training boundless loop.
Pantelis Monogioudis: That, does a forward pass.
Pantelis Monogioudis: That relates an error.
Pantelis Monogioudis: And, trains with, you know, the network with,
Pantelis Monogioudis: the power variation operation. This is the composition of the losses, right? Of the predictions of all the context worlds that are around you, as I discussed. You've got a loss, and you can actually see, at the end of the day, you obtain
Pantelis Monogioudis: W matrix, W star matrix, that it's going to, sort of, be used for
Pantelis Monogioudis: picking up the embeddings. You can also produce the embeddings of the vocabulary, and actually you can upload it, if I remember right, you can upload it.
Pantelis Monogioudis: Anyway, it is a… it is a trivial time of implementation from scratch of the world, like…
Pantelis Monogioudis: There's also another kind of non-front-scratch implementation, if you want to take a look on there. We've got a notebook for Pytes. I just didn't have time to link that here.
Pantelis Monogioudis: Alright.
Pantelis Monogioudis: So that's it, basically… yes, go ahead.
Pantelis Monogioudis: I had a question regarding the vocabulary used in the embedding model. Yes. So, how do we ensure that the vocabulary that we're using for the embedding model, that we're training for the embedding model, is the same as the one that we'll be using in downstream tasks? So, for example, it could happen that a word that wasn't seen in a… during the training of the embedding model.
Pantelis Monogioudis: is encountered when we are actually using the… So tokenizers and the models are kind of jointly developed. They are linked, effectively, right?
Pantelis Monogioudis: So you have a tokenized, you have a vocabulary, and then you develop a language model that's compatible to that vocabulary.
Pantelis Monogioudis: So this fact that if one of the words is unknown, for example, one of the sub-words is completely unknown, let's just say you didn't follow that kind of rule, and there's lots of unknowns, right, that's not good for prediction for the language model, because you're going to have a special token.
Pantelis Monogioudis: which is going to be, assigned every time you have an unknown subord, right? If you have many of them, that's not good.
Pantelis Monogioudis: You will have some, but, definitely, not, shouldn't have many.
Pantelis Monogioudis: Yes. If, if I'm not, words.
Pantelis Monogioudis: deteriorate the performance. Then, how do LLMs handle, typos, or, like, spelling mistakes in prompts? Okay, so,
Pantelis Monogioudis: How do they do, spelling mistakes?
Pantelis Monogioudis: Okay, I'm not 100% sure, if they, internally, how a spell checker will work, right? But even before then, they had the spell checking capabilities. It's, like, probably a statistical. I'm not 100% sure how the spell checking will work, but I can, I can, I can come back to you on that.
Pantelis Monogioudis: But definitely the… the…
Pantelis Monogioudis: sort of, the ability of, I mean, the possibility of finding kind of unknown sports is greatly reduced with it by parent coding.
Pantelis Monogioudis: Okay. But if you on purpose have something, that it is not, it is, it is not, aligning, if you like, with the proper word, let's say you misspelled something, right? Then this word most likely will consist of,
Pantelis Monogioudis: second… sub-words, and yeah, I need to come back to you, I do not know the answer.
Pantelis Monogioudis: How they would do the correction.
Pantelis Monogioudis: We will see a method a bit later on where
Pantelis Monogioudis: Language models, typically, they don't predict only the next token, but they predict the maximum likelihood sequence of tokens.
Pantelis Monogioudis: So… so they go back and correct, effectively, previous predictions. Obviously, they don't correct
Pantelis Monogioudis: immediately, in the sense that they have to wait for the sequence, or some sequence, some window, and then they calculate the maximum likelihood of a sequence. It's a MLSC, that's also known as Viterbi decoding, and they go and say, okay, the total maximum like probability, or the lack of probability of the whole sequence is now larger than the one that I predicted initially.
Pantelis Monogioudis: So they go back and replace that graph, but what you see is the corrected bench.
Pantelis Monogioudis: So something like that may be happening in the auto-correction space, so… Alternatively.
Pantelis Monogioudis: But I think that's been making a mistake.
Pantelis Monogioudis: All right, so one thing that I wanted to now start a little bit of discussion is to see how we are building some kind of a trivial language model.
Pantelis Monogioudis: That… that are going to,
Pantelis Monogioudis: that I'm going to be constructed using,
Pantelis Monogioudis: So the next step is we will take the embedding, right, and we will be creating language models.
Pantelis Monogioudis: That, will take the following force. P-model.
Pantelis Monogioudis: or WT given WT minus 1 comma WT minus 2, comma, got a low WT plus minus N.
Pantelis Monogioudis: Tomasija.
Pantelis Monogioudis: So over here, the conditional probability distribution is different.
Pantelis Monogioudis: And this is basically what we'll be calling context.
Pantelis Monogioudis: This context could be a very large number these days, but initially it turned out to be pretty small numbers.
Pantelis Monogioudis: At this stage is actually perfectly fine context associated with, you know, 100,000 entries there.
Pantelis Monogioudis: Probably than larger sometimes.
Pantelis Monogioudis: And, so the language model is a model that predicts the next token, given…
Pantelis Monogioudis: The previous sequence of tokens.
Pantelis Monogioudis: And it's parametric model, obviously, will adopt exactly the same kind of criterion for optimizing the theta, and will match
Pantelis Monogioudis: the, again, the P data distribution present in my training. So we'll be building some models using some neural networks, so the… how do you… how do you come up with P models? Evidently, you need to…
Pantelis Monogioudis: have neural architectures that are, I would call them, Pretty friendly.
Pantelis Monogioudis: And, with sequences.
Pantelis Monogioudis: Okay, so you need to construct some architectures which are, sort of inherently have, this kind of notion of the T-index now, and this, this T-index will be changing some internal representation that they are building, will be calling a state.
Pantelis Monogioudis: So, we'll discuss this architecture, which is called a current.
Pantelis Monogioudis: New warehouse?
Pantelis Monogioudis: Networks.
Pantelis Monogioudis: And then we'll, develop the transformers as
Pantelis Monogioudis: changes in this RNN kind of architecture we will be introducing.
Pantelis Monogioudis: No, let's, discuss a little bit about them.
Pantelis Monogioudis: For example, Chuck.
Pantelis Monogioudis: So, just like what we have seen, kind of, in the past.
Pantelis Monogioudis: We had, from the discussion of the, Kalman filter, we had this notion of a state, S of T,
Pantelis Monogioudis: which is, XYZ, let's say, B, you're all… This is Vince.
Pantelis Monogioudis: V, and V dot, if you remember that. That's basically a notion of a state of a vehicle, for example. So we will maintain this kind of state notion.
Pantelis Monogioudis: And we will be constructing now, not a… we will not be,
Pantelis Monogioudis: Making a lot of assumptions about that kind of state.
Pantelis Monogioudis: dependency to other things. Before, in the hidden Markov model, we made assumptions like this, and the probabilistic graphical model had a very special structure, right? So here, we will build,
Pantelis Monogioudis: another kind of capability, where we will be modeling the S of T that, in general can be
Pantelis Monogioudis: associated with SD-1.
Pantelis Monogioudis: and the action kind of A of T.
Pantelis Monogioudis: Via an equation, That is called the… let me call this, fully… dynamic, model.
Pantelis Monogioudis: Because even that, function F,
Pantelis Monogioudis: That we made it first in our following kind of diagram, the target kind of function, changing over… changing over time, changes over time.
Pantelis Monogioudis: We are not going to be able to model such functions.
Pantelis Monogioudis: What we are going to be modeling, however, here is something that does not change over time. The function itself does not change over time. The mapping, in other words, from X to the Y is not changing over time.
Pantelis Monogioudis: So we're remodeling something that it will, be, this…
Pantelis Monogioudis: And before, while in the Karman filter we had the explicit actions that the agent was taking.
Pantelis Monogioudis: an action, let's say, as I move forward.
Pantelis Monogioudis: Over here, the action will be implicit. The action could be the arrival of the next
Pantelis Monogioudis: Sub-word in my input.
Pantelis Monogioudis: Right? It could be an event.
Pantelis Monogioudis: That, that I'm going to, consider as an action.
Pantelis Monogioudis: So our models will actually look now
Pantelis Monogioudis: So this is basically, impossible.
Pantelis Monogioudis: two models.
Pantelis Monogioudis: And, this, this will be, you know, this is basically what… We…
Pantelis Monogioudis: Our modeling in this lecture.
Pantelis Monogioudis: And similar to what we are, going to…
Pantelis Monogioudis: did in the past. We will start with,
Pantelis Monogioudis: Refreshing, if you like, our memory with respect to the sigmoidal kind of neuron.
Pantelis Monogioudis: And we'll arrive on a neuron that it is,
Pantelis Monogioudis: Recurrent, as we call it. And this neuron,
Pantelis Monogioudis: Has, as elements, everything that we see in this kind of reparation.
Pantelis Monogioudis: So this equation will change our notation when we are representing this. This H of T will become a hidden state, H of T. Therefore, I'm going to use the letter H.
Pantelis Monogioudis: This, F would be replaced by my hypothesis function, the G, as we call it always. HD-1 is there.
Pantelis Monogioudis: the arrival… of a…
Pantelis Monogioudis: the next point in the sequence, could be a number, could be something else, is going to be denoted by X of t.
Pantelis Monogioudis: And obviously, I have A parametric hypothesis by using the letter theta.
Pantelis Monogioudis: So this is going to be our model.
Pantelis Monogioudis: We need something.
Pantelis Monogioudis: Before, Listening was missing.
Pantelis Monogioudis: Yes.
Pantelis Monogioudis: And the second one? Yeah. The difference is that the target function is time-value.
Pantelis Monogioudis: In the first one. It's impossible to monitor. And this one is the same? This one is assumed to be the same. It won't change function.
Pantelis Monogioudis: Obviously, this is not only introducing non-stationary… non-stationary in our program. It's, chaotic.
Pantelis Monogioudis: Over here, we obviously, we can have non-station, I mean.
Pantelis Monogioudis: non-sectionality by the fact that my PD data heart may be changing.
Pantelis Monogioudis: Over time, right? But typically, we do not assume that we have distributional drifts in any of the problems. Of course, in actual systems, you have to account for drifts, right? But over here in our class, we don't assume we have drifts.
Pantelis Monogioudis: Okay, all right, so let's see how the simultane neuron will be replaced now with this recurrent neuron. So we will be maintaining the dot product.
Pantelis Monogioudis: So, H of D is coming in.
Pantelis Monogioudis: We will be maintaining the data product between
Pantelis Monogioudis: a parameter that we call U. We used to use W before, so W transpose X was there with us, now it's going to be called U because we'll use the W for something else.
Pantelis Monogioudis: We had some kind of a bias addition.
Pantelis Monogioudis: And, instead of a sigmoidal unit, I'm gonna…
Pantelis Monogioudis: I use a tonnage, a hyperbolic tan function, That will give me… the hidden state.
Pantelis Monogioudis: And, the hyperbolic time function looks very similar to the sigmoidal.
Pantelis Monogioudis: function, but… is not, and I'll explain why
Pantelis Monogioudis: the place with the damage function, why it made some sense.
Pantelis Monogioudis: So, what, ask the equation?
Pantelis Monogioudis: As the question implies, we need to consider what came… what output we formed before.
Pantelis Monogioudis: That i1 of X of t. So we have to store in a memory, this is memory.
Pantelis Monogioudis: in,
Pantelis Monogioudis: in logic, we have this kind of deep flip-flop to store stuff in there. So, obviously, we are storing the previous, hidden state, HD-1, but that we can use this memory cell to retrieve it.
Pantelis Monogioudis: So, when we retrieve it, We'll adjust it by… Small letter W.
Pantelis Monogioudis: And we are going to… Dude.
Pantelis Monogioudis: I'm just… But… Before they don't imagine.
Pantelis Monogioudis: So if you want to write the equation, He's going to be… H of D.
Pantelis Monogioudis: Is it while tanage?
Pantelis Monogioudis: of U transpose X of T, Class WHT-1, Last week.
Pantelis Monogioudis: So, if you contrast this guy, And this guy…
Pantelis Monogioudis: you can actually see now that the G function is… a non-linear, function of… these steps.
Pantelis Monogioudis: Is the hidden state a vector?
Pantelis Monogioudis: He's a hidden state, isn't it?
Pantelis Monogioudis: Nope.
Pantelis Monogioudis: Because this is a scalar, this is a scalar, this is a scalar, therefore this is a scalar. So the only thing that comes out here is a scalar, just like the sigmoidal neuron.
Pantelis Monogioudis: We don't impose any probabilistic interpretation of this Fame, but this is coming.
Pantelis Monogioudis: So what are the trainable parameters here of beta?
Pantelis Monogioudis: Greenable parameters are… the U vector.
Pantelis Monogioudis: The feed bias?
Pantelis Monogioudis: And, W.
Pantelis Monogioudis: Okay.
Pantelis Monogioudis: Fair enough. So what we have achieved here, we have achieved something that, could be very well
Pantelis Monogioudis: Potentially very well, allow us to model a time series.
Pantelis Monogioudis: So forget about language modeling, just focus a little bit on this kind of problem of time series modeling. I think there's a lot of time series prediction around here in this neighborhood.
Pantelis Monogioudis: And, lots of people are getting lots of money for predicting the right market and stuff like that, based on time series kind of data.
Pantelis Monogioudis: But, as you can understand, we don't have a lot of hopes about this guy. We don't have a lot of hopes, but let's try to use it in a problem which is a time series kind of problem. So, I'm just giving you now…
Pantelis Monogioudis: This problem will actually be used to
Pantelis Monogioudis: discuss something which is kind of essential. So I'm gonna use some kind of, time series.
Pantelis Monogioudis: I think in your midterm, you had some time series as well, right? And, this was,
Pantelis Monogioudis: Let's assume that I am… giving you… A time series like this.
Pantelis Monogioudis: And I'm asking you, to predict.
Pantelis Monogioudis: The next… Member of the sequence.
Pantelis Monogioudis: So, the next member of the sequence… is… Has a ground roof.
Pantelis Monogioudis: But it's next 50.
Pantelis Monogioudis: Okay, I'm giving you… X0 to X49, and I'm asking about the X15.
Pantelis Monogioudis: So, the problem… Is to predict the price… of, a commodity.
Pantelis Monogioudis: In the next… Trading.
Pantelis Monogioudis: Nope.
Pantelis Monogioudis: interval.
Pantelis Monogioudis: from…
Pantelis Monogioudis: a sequence, of capital G is equal to 50, earlier.
Pantelis Monogioudis: Crisis.
Pantelis Monogioudis: you know, we can actually solve this problem without RNNs or not.
Pantelis Monogioudis: How did you solve your problem in the midterm?
Pantelis Monogioudis: You didn't? Snapshots? A particular time. Well, in the midterm, most probably many of you will probably consider some wonderful window.
Pantelis Monogioudis: Right. Where samples are coming in, in this kind of window. This would be the window that I can use as input, if I convert it to a vector, to feed it into a DNN, right? And then the output of the DNN
Pantelis Monogioudis: will actually be, whatever the problem was asking you. I think the output was some kind of activity detection, or something like that.
Pantelis Monogioudis: Okay, over here, Let's assume that,
Pantelis Monogioudis: We have this kind of sequence, and let us show that there's someone asking to do the next training kind of interplay.
Pantelis Monogioudis: I think very important to understand here is that our prediction
Pantelis Monogioudis: It's going to be very poor.
Pantelis Monogioudis: From, observing a time series such as this.
Pantelis Monogioudis: And if I told you, if I… especially in a kind of machine learning, obviously you can do a lot of things in a kind of a classical time series kind of modeling. ARMA, MA, whatever, you know, all these kind of algorithms come to mind, but when it comes to training something, to predict
Pantelis Monogioudis: Do we need, one example, or…
Pantelis Monogioudis: many examples.
Pantelis Monogioudis: And the answer is many examples. So, someone would say, hold on a second, I have an example here of this guy followed by this guy. I have this guy followed by this guy. But over here, I'm asking you to say, okay, consider all the 50, right, earlier, and give me this, right?
Pantelis Monogioudis: So you can actually build some form of intelligence and kind of windows, you can, especially if you have a much longer kind of time series, you can actually go and produce many of these kind of examples. And this is basically what it should happen. Out of just one sequence, it's very tricky.
Pantelis Monogioudis: you will get very poor results if you have just a prediction problem based on one example. So I want you to consider this time series as one example. Just like in computer vision, we have many examples of dogs that we wanted to see, to train, to come up with the concept of a dog.
Pantelis Monogioudis: In a similar fashion here, we need many of these series to…
Pantelis Monogioudis: B2c, many of this kind of series, with some kind of ground proof, to build a model that will do their prediction.
Pantelis Monogioudis: So, I think that…
Pantelis Monogioudis: So, so the, the, the training, so the conclusion of this discussion, so the training data.
Pantelis Monogioudis: Must be.
Pantelis Monogioudis: M… serious.
Pantelis Monogioudis: off, this… commodity.
Pantelis Monogioudis: M series of this kind of commodity, with ground tools.
Pantelis Monogioudis: So we have M of this guy, M of guys. What I'm trying to capture here…
Pantelis Monogioudis: What would be my… output.
Pantelis Monogioudis: And how this is related to this kind of recurring kind of community.
Pantelis Monogioudis: So, Derekara, you're not being,
Pantelis Monogioudis: Producing a scalar is very capable of solving this problem if it feeds… if we feed with many of these series.
Pantelis Monogioudis: We create, if you like, some error, right, between the actual prediction and the ground group kind of prediction, and we, in a typical fashion, drive an optimization algorithm to change the training kind of parameters to minimize that error.
Pantelis Monogioudis: Okay, that error is obviously a well-known error class, convey square.
Pantelis Monogioudis: Okay, it's an aggression problem.
Pantelis Monogioudis: So, what we will do, is we'll do that.
Pantelis Monogioudis: Let's assume that we do that.
Pantelis Monogioudis: Okay, here it is.
Pantelis Monogioudis: So… I'm, getting this kind of, M of this kind of, time series.
Pantelis Monogioudis: And, I am going to construct a simple RNN.
Pantelis Monogioudis: Waited.
Pantelis Monogioudis: Here it is, a single kind of RSN. Just exactly the same block diagram we have just through, that, one here represents the number of neurons that we have, the current neurons. We'll see that we need to increase this number, that's the discussion we're going to have now.
Pantelis Monogioudis: To do the training. Okay, that's basically our simple neuron. We do the… some casin of STD here with some kind of learning rate, and we are training this kind of thing to predict
Pantelis Monogioudis: To predict the next, point in that kind of sequence. That's the problem here. And we… this thing is kind of converging fine. And, finally, we found ourselves in the following kind of situation. If you see the min-square error of this guy, and you go back and do the same thing with a dense layer.
Pantelis Monogioudis: You will find that a densely performs better than this.
Pantelis Monogioudis: And what could be, potentially, the reason for that? To save you some time of thinking?
Pantelis Monogioudis: If you… See the number of trainable parameters.
Pantelis Monogioudis: In the dense layer, which will be…
Pantelis Monogioudis: 50, plus the biased terms, right? Because you get the whole sequence.
Pantelis Monogioudis: through that kind of a dense matrix, right, so through the kind of a dot product, right, to… and use 50 numbers, right, to learn in that kind of a dense kind of setting. Over here.
Pantelis Monogioudis: The trainable parameters that we're going to have with the type of a single neuron is There's no you.
Pantelis Monogioudis: It's just one at a time is coming in. There is B, which is another scalar, and another scalar, so we have 3. So we're comparing 3 versus 51 each time.
Pantelis Monogioudis: So you have, apples and all this kind of comparison.
Pantelis Monogioudis: If you are… Take this low diagon, which has this kind of, feedback kind of path.
Pantelis Monogioudis: In it, and you want to unroll it. You want to remove the feedback connection and see what's really happening. You will arrive in this,
Pantelis Monogioudis: Easy-to-remember kind of diagram, where… over here.
Pantelis Monogioudis: At each point in time, an input arrives, A hidden state is calculated.
Pantelis Monogioudis: And the next leap happens over and over again.
Pantelis Monogioudis: Let me call it RNN1.
Pantelis Monogioudis: RNN1 is executed all the time.
Pantelis Monogioudis: And then finally, When we arrive at the… reduce the age 50,
Pantelis Monogioudis: This page 50 is our white hat.
Pantelis Monogioudis: Which is obviously going to be leading itself to a loss function.
Pantelis Monogioudis: That, will… this loss function is going to get my… as input the X50.
Pantelis Monogioudis: To do it safe.
Pantelis Monogioudis: Because we are.
Pantelis Monogioudis: It is this equation.
Pantelis Monogioudis: Biscuiting equation H2 is created from looking at the new input that arrives, Questioning, age swelling.
Pantelis Monogioudis: H3 needs H2. H4 needs H3. So there's a sequential processing that is actually happening.
Pantelis Monogioudis: It's very important to… think the unrolled kind of architecture as, effectively, a much deeper
Pantelis Monogioudis: network, trivial network in this kind of case, than what we originally kind of
Pantelis Monogioudis: thinking, as they were originally kind of imagining. So… so this is a…
Pantelis Monogioudis: Effectively, sequential processing of, applying exactly the same equation given a new input, until we reach our destination.
Pantelis Monogioudis: And so, from that kind of perspective, obviously.
Pantelis Monogioudis: There is a following kind of problem, which we'll discuss a little bit later, about buffer by gating through this kind of chain.
Pantelis Monogioudis: That's a… that's a kind of a… seems to be a problem with 50 stages here. But that's why, let's put that kind of aside. The next problem we actually discuss is, is this enough?
Pantelis Monogioudis: Or… This prediction even problem, because despite the fact that you gave us 10,000 of these guys.
Pantelis Monogioudis: Let's assume, and the notebook is using 10,005 of this kind of sequences.
Pantelis Monogioudis: There's something that is not really correlating with our intuition. For example, if this commodity was the…
Pantelis Monogioudis: price, let's say, of the Tesla stock, okay?
Pantelis Monogioudis: The question I have is that can one hidden state
Pantelis Monogioudis: which is, in this case, is progressively determined by looking at this sequence of 50 values. Can one hidden state represent all the information needed for
Pantelis Monogioudis: capturing the latent factors that can affect the stock price.
Pantelis Monogioudis: And the answer is very unlikely, unless the stock price is trivial, and then the dynamics are trivial, and then probably will not be traded.
Pantelis Monogioudis: There must be some kind of more than one underlying latent factors that drive the stock price. For example, one factor would be associated into it, again, this is not something which happens in reality, but
Pantelis Monogioudis: we can associate one variable, let's say Z, to capture, or sorry, one, one variable, let's say H1, to capture the, sort of macroeconomic situation, by some macroeconomic indicator.
Pantelis Monogioudis: Another one to capture, you know, pricing policy, or whatever, for revenue, gas flow, whatever kind of drives the prices of the stock.
Pantelis Monogioudis: So the conclusion is that we need… this guy.
Pantelis Monogioudis: If we want to create a representation of the
Pantelis Monogioudis: internal dynamics of that kind of a stock, right? This guy has to be available.
Pantelis Monogioudis: And… and it's kind of a… requires kind of some thinking to actually arrive at this kind of a natural kind of extension.
Pantelis Monogioudis: Obviously, we have seen this accession before. We went from a single sigmoidal neuron to more than one, to define a layer, right? The dense layer. We are going to go into the same direction. We're going to start with a single recurrent neuron, and we'll arrive into the RNN layer.
Pantelis Monogioudis: But the intuition is that, we want, in many problems, the hidden state to involve more than one.
Pantelis Monogioudis: But no one.
Pantelis Monogioudis: That is… that is the… that is the intuition. I tried to explain the intuition, but, with whom?
Pantelis Monogioudis: with the stock price. Okay, so let's, let's kind of write things down.
Pantelis Monogioudis: So, first thing that we've seen is that, We need to increase.
Pantelis Monogioudis: The number of parameters.
Pantelis Monogioudis: But the second thing is not, this increase in the number of parameters is okay, I mean, it's kind of trivial, okay, fine.
Pantelis Monogioudis: But still, we have a price prediction problem.
Pantelis Monogioudis: And so, if H of T is a vector.
Pantelis Monogioudis: How we are going to get the price out of it.
Pantelis Monogioudis: So, we need to decouple, the hidden state, From the particular value.
Pantelis Monogioudis: We need to do this kind of decoupling, right? And we were familiar on how to do this decoupling from previous kind of discussions, but let's write it down to make sure that we have it. So, we need…
Pantelis Monogioudis: So… make.
Pantelis Monogioudis: the hidden state…
Pantelis Monogioudis: as Ector.
Pantelis Monogioudis: that we'll call H of T again.
Pantelis Monogioudis: And, at the same time, The couple?
Pantelis Monogioudis: Oh, eat.
Pantelis Monogioudis: Chrome.
Pantelis Monogioudis: The dimensionality.
Pantelis Monogioudis: of… the target variable?
Pantelis Monogioudis: Why?
Pantelis Monogioudis: Y has one dimension?
Pantelis Monogioudis: Obviously, the hidden state here has…
Pantelis Monogioudis: More than one. We'll have more than one. And of course, the next kind of interesting question is, okay, how many?
Pantelis Monogioudis: what is the dimensionality of the hidden state for a problem? And the answer is…
Pantelis Monogioudis: We do not know for sure, but we have some way, potentially, to estimate it. And, we can make some, rough estimate.
Pantelis Monogioudis: Using your, famous, last question of your metrics.
Pantelis Monogioudis: Which had to do with, expressing the covariance matrix of a random variable in a recursive kind of way.
Pantelis Monogioudis: And it's not really the recursion that we will be using here, but again, the intuition is the following. If I give you
Pantelis Monogioudis: a capital X matrix, which I call now the data matrix, like, almost like a parenthesis.
Pantelis Monogioudis: if I give you a capital X matrix that has some form of,
Pantelis Monogioudis: Dimensions, let me call them, N?
Pantelis Monogioudis: And some kind of, data enters here, let's call it M.
Pantelis Monogioudis: I think I use the same kind of letters.
Pantelis Monogioudis: And I want to ask you the following question.
Pantelis Monogioudis: How many of these dimensions do I really need?
Pantelis Monogioudis: to model.
Pantelis Monogioudis: What would be yours?
Pantelis Monogioudis: Because at the essence, we are calculating covariance matrices for answering similar questions, such as this.
Pantelis Monogioudis: I will close with this question.
Pantelis Monogioudis: How many?
Pantelis Monogioudis: What will you… you've been now through this kind of seven lectures, and this is the interview question. How many dimensions? What should I do to select
Pantelis Monogioudis: To see that this matrix may contain some form of redundancies here.
Pantelis Monogioudis: dimensionality reduction. I mean, that's basically where I'm going to do dimensionality reduction, and I'm going to have a criterion that says, okay, if I…
Pantelis Monogioudis: don't need it.
Pantelis Monogioudis: All of these kind of dimensions represent how my X random of variable, not the data matrix, behind the data matrix is something… some X that is changing, right? The X is one row of this guy, right?
Pantelis Monogioudis: So, another row, M generating another row, another row, so all the way to M rows. What is really the latent variables that are worth preserving from this small letter X?
Pantelis Monogioudis: So, you can do a simple linear projection.
Pantelis Monogioudis: a principal component analysis, in fact. In fact, you will do SVG.
Pantelis Monogioudis: If you do an SVD, a singular value decomposition, you will be able to say, okay, if I… I will see the spectrum of my principal values, and if my spectrum of my principal values most likely will be something like this.
Pantelis Monogioudis: Something that is… has strong values in the first, second, and third dimension, but not so strong values in the fourth and fifth dimension.
Pantelis Monogioudis: And you can say that I can live without the fourth and fifth dimension.
Pantelis Monogioudis: And I can live with only 3 out of the 5 dimensions.
Pantelis Monogioudis: Similar thinking you can actually do with…
Pantelis Monogioudis: the stock price of a commodity, you can start aggregating the stock price, and you can pick up
Pantelis Monogioudis: Because these are all public information, all the features of the stock.
Pantelis Monogioudis: the P over E value, whatever have you, cash flow, blah blah, and so on, you can put it in this gigantic table, X.
Pantelis Monogioudis: And you can say, okay, I have this kind of data, how can I select here what is really the most important factors?
Pantelis Monogioudis: These most important factors in the SVD domain, in the PCA domain, will be called latent factors, or hidden factors, or hidden state that we had over there. It's just that the hidden state that we have is itself time varying.
Pantelis Monogioudis: depending on how many inputs we are considering, right? So this number of rows are…
Pantelis Monogioudis: Increasing and increasing and increasing, and…
Pantelis Monogioudis: But you cannot really consider everything from the beginning of today, from today all the way until the company is founded, right? You will find inside there what? Non-stationary behaviors, right? So you need to limit yourself within a window, right?
Pantelis Monogioudis: So, before he sends the tweet, and then after he sent the tweet, that's basically the point of disc… disconnecting this kind of price from, you know, having non-stationary kind of decaders, right? So that's basically an intuitive way where I can potentially start designing the number of dimensions I need for my hidden state.
Pantelis Monogioudis: So, next, time, we will expand the discussion to have,
Pantelis Monogioudis: a discussion associated with, how to increase the number of parameters, so we will create the RNN kind of layer, and close the discussion with, understanding what is memory.
Pantelis Monogioudis: inside this kind of structure. What we say when we say, okay, we need to have some memory in a… during a kind of a learning process, it will also be connected to the concept of gradient flow.
Pantelis Monogioudis: Do you have my, piece of paper so I can leave? Yes? Who has not signed it yet? Yes, obviously, yes, go ahead.
Pantelis Monogioudis: Yeah, because then we'll learn about film.
Pantelis Monogioudis: They were likely to deal and deliver those.
Pantelis Monogioudis: Correct.
Pantelis Monogioudis: But this is maybe a bit more general than it does, right?
Pantelis Monogioudis: Mind you that this is,
Pantelis Monogioudis: It's not exactly next to each other, but in the context.
Pantelis Monogioudis: Yeah, okay, now I understand what you're saying. So you can say, for example, take an order today, and you know, most likely, this type of bolts
Pantelis Monogioudis: probably are not necessarily so conceptual. For example, like I said, you can have a car, a full store, but then you can have the car store, and also be there. So you will have, some…
Pantelis Monogioudis: notion, like… it could be…
Pantelis Monogioudis: the car, or it could be something else. I don't know how, if I can sort of fresh, say with 70 that the car will always be the highest. Yeah, my conclusion was just that we are expecting the world closer to each other in the snacking sense.
Pantelis Monogioudis: Like, it doesn't make sense to, like, not have a single day, because the closest embedding would be the same building, right? Yeah, and I think when you show the projection, I think you had the parts…
Pantelis Monogioudis: It's definitely the paper.
Pantelis Monogioudis: Okay, so I would say that, maybe some, notes from, the Stanford University course that I also linked in the website.
Pantelis Monogioudis: But definitely there is the reverse, where we are trying to predict the center move, they're positive. That is called the bottom of words. And, this is a mirror construction of the same character. I think, obviously, there are landing from the… And, of course, this is how things are okay.
Pantelis Monogioudis: For me, understanding the concept of a business.
Pantelis Monogioudis: So I looked up, the…
Pantelis Monogioudis: Basically, I might have to change the, environment command, of course, to, allow the programmer to go somewhere.
Pantelis Monogioudis: you have to have the port, yeah, there, and the port, you know, the purpose, I think, doesn't something. Yeah. So, I mean, I'd like to change the first 5 days, and if I do that, then I'd like to remove the container. Well, I was able to run even outside is okay, could run on the command line of the laptop. You don't even have to open.
Pantelis Monogioudis: So, by setting the network, it will host any document, container, one terminal in my local… definitely have a local run sound model, right? And then, the doctor will be able to connect…
Pantelis Monogioudis: Okay. Yeah.
Pantelis Monogioudis: Yeah. It's fair. I have literally full reading yesterday.
Pantelis Monogioudis: They haven't sent a confirmation. Or even if they did, they're just… So…
Pantelis Monogioudis: Bye.
Pantelis Monogioudis: No plastic.
Pantelis Monogioudis: I was wearing them during summer, and I…
Pantelis Monogioudis: I didn't think it was actually good.
Pantelis Monogioudis: Oh my god. Yeah.
Pantelis Monogioudis: Bye-bye.
Pantelis Monogioudis: Oh, we're scared.
Pantelis Monogioudis: Question.
Pantelis Monogioudis: Because the end user for CS or robotics player, that's needed.
Pantelis Monogioudis: Lord.
Pantelis Monogioudis: Financially, you mentioned.
Pantelis Monogioudis: Hello, my uncle.
Pantelis Monogioudis: Hmm. Were you deceiving them?
Pantelis Monogioudis: I don't see any recordings, no.
Pantelis Monogioudis: We're exhaustive.
Pantelis Monogioudis: That's just the way it… Would there any other…
Pantelis Monogioudis: And you're key for the hood.
Pantelis Monogioudis: We all should be…
Pantelis Monogioudis: Bye.
Pantelis Monogioudis: Premier Sinho.
Pantelis Monogioudis: They're gonna be both familiar.
Pantelis Monogioudis: You do.
Pantelis Monogioudis: that targeted.
Pantelis Monogioudis: You see, you see control of the Shamania, you see the O, you see areas, you can come to the community of that. You see there.
Pantelis Monogioudis: Is it okay with…
Pantelis Monogioudis: So I did… So… I know you have a different one.
Pantelis Monogioudis: Hopefully, you receive an alternative. If you want to get onto the skiing.
Pantelis Monogioudis: I think there are two.
Pantelis Monogioudis: Thor… mein.
Pantelis Monogioudis: I think it's about the end.
Pantelis Monogioudis: What?
Pantelis Monogioudis: I'll get a little genius react.
Pantelis Monogioudis: That's how the word of CMU.
Pantelis Monogioudis: We'll miss the number half.
Pantelis Monogioudis: The image was the CMU.
Pantelis Monogioudis: Also, obviously, you mentioned either USC, and you said.
Pantelis Monogioudis: academics or a cheerle.
Pantelis Monogioudis: CMUCK Robotics, or CS Watch, and you miss the Ace, obviously. But, at the same time, you wish to hear Acehi.
Pantelis Monogioudis: That'll be decisions.
Pantelis Monogioudis: stealing your college year.
Pantelis Monogioudis: So…
Pantelis Monogioudis: But I'm a robotics resident, CMU, obviously, best year.
Pantelis Monogioudis: You can do what you're dying.
Pantelis Monogioudis: What the paper decided?
Pantelis Monogioudis: should design…
Pantelis Monogioudis: You mentioned that?
Pantelis Monogioudis: Acha, megones.
Pantelis Monogioudis: So…
Pantelis Monogioudis: back of SEO.
Pantelis Monogioudis: Okay, Edie, when I was CME wasn't there. But TME program, Oh, wake up.
Pantelis Monogioudis: That's it. Tao.
Pantelis Monogioudis: I don't think that post figure says that as much, right?
Pantelis Monogioudis: But they're obviously more than…
Pantelis Monogioudis: You, Bish.
Pantelis Monogioudis: Or body, like, all of them.
Pantelis Monogioudis: If you open to, obviously, higher ed school and higher ed, you can see coursework, obviously, ranking at CMU or at CAUST.
Pantelis Monogioudis: But…
Pantelis Monogioudis: Location for sidewalk. Location CM near Yumeshka, same here. Okay? Almost here.
Pantelis Monogioudis: or a CS.
Pantelis Monogioudis: Per meconic, although… to the CMU Robotics companies at OPCA robotics Company, Michigan.
Pantelis Monogioudis: So… So, what is the other major attachment is not that?
Pantelis Monogioudis: We don't want to accept any of this.
Pantelis Monogioudis: Bob is!
Pantelis Monogioudis: go to with bases on Votex.
Pantelis Monogioudis: mein.
Pantelis Monogioudis: To ensure there was a male kind of professor, so I want to work with him.
Pantelis Monogioudis: Cool.
Pantelis Monogioudis: And I need to do…
Pantelis Monogioudis: aka to
Pantelis Monogioudis: But…
Pantelis Monogioudis: That's what it is.
Pantelis Monogioudis: I see a financial transaction.
Pantelis Monogioudis: You're using the internet.
Pantelis Monogioudis: Give me this association.
Pantelis Monogioudis: How much do you go there? Yeah?
Pantelis Monogioudis: I don't put your data.
Pantelis Monogioudis: As you can tell I'm a lot of people.
Pantelis Monogioudis: UTL.
Pantelis Monogioudis: I… I am excited.
Pantelis Monogioudis: 27…
Pantelis Monogioudis: Yeah.
Pantelis Monogioudis: Did you have enough?
Pantelis Monogioudis: Maybe you're gonna be one of these.
Pantelis Monogioudis: Oh, why do you watch for?
Pantelis Monogioudis: Funny!
Pantelis Monogioudis: Or just the right, as it's a happy chapter is about perfection.
Pantelis Monogioudis: The aggressive.
Pantelis Monogioudis: Which looks tiny, but…
Pantelis Monogioudis: Are you convincing that?
Pantelis Monogioudis: Can we have, that's a good transfer because of SEM machine, technically.
Pantelis Monogioudis: There we go see America, pretenders.
Pantelis Monogioudis: Who gave my actual gifts, huh?
Pantelis Monogioudis: I can't remember.
Pantelis Monogioudis: I actually thought.
Pantelis Monogioudis: I just throwing the energy, so I'm gonna, like, throw my ear.
Pantelis Monogioudis: Honestly.
Pantelis Monogioudis: Overly.
Pantelis Monogioudis: Fair.
Pantelis Monogioudis: Thank you, sir, so much.
Pantelis Monogioudis: I chose as I don't like fighting in.
Pantelis Monogioudis: They go very fast deal.
Pantelis Monogioudis: CM, you've got chip design, CM, you've got robotics Player, CM you've got CN, Yumich got finance, CM researcher.
Pantelis Monogioudis: or landing photo opara, you may be a tarnish anymore.
Pantelis Monogioudis: Diga?
Pantelis Monogioudis: Yeah.
Pantelis Monogioudis: Stay in there, and then bye.
Pantelis Monogioudis: Jewish don't right there, huh?
Pantelis Monogioudis: Or engineering degree top 6.
Pantelis Monogioudis: You know what I mean?
Pantelis Monogioudis: So, these campaigns… karah.
Pantelis Monogioudis: We have a laptop on there, but I agree.
Pantelis Monogioudis: las volado.
Pantelis Monogioudis: Do you need any amended?
Pantelis Monogioudis: Nice!
Pantelis Monogioudis: Lord, I think.
Pantelis Monogioudis: You're done.
Pantelis Monogioudis: So, I still love to see me a black and then?
Pantelis Monogioudis: I mean, those of the CMU have, obviously, you must say.
Pantelis Monogioudis: But… would have benefited them.
Pantelis Monogioudis: Boy, it was… Otherwise, it will be here.
Pantelis Monogioudis: Honestly, I'm not sure when we…
Pantelis Monogioudis: Responding that?
Pantelis Monogioudis: Good evening.
Pantelis Monogioudis: What were you thinking about that push out the better at Zoom?
Pantelis Monogioudis: We have anything to do with that.
Pantelis Monogioudis: We should… Sean Apple.
Pantelis Monogioudis: 2019.
Pantelis Monogioudis: Agent.
Pantelis Monogioudis: Good morning, Bob, singles.
Pantelis Monogioudis: Shit.
Pantelis Monogioudis: Energy.
Pantelis Monogioudis: terms of privilege at CMU's after the CMU data.
Pantelis Monogioudis: After EDC, you're gonna have EDU mission, right? ED has…
Pantelis Monogioudis: dihan.
Pantelis Monogioudis: kernel push layer, you see a raised spot in Kernel.
Pantelis Monogioudis: Life's coming by.
Pantelis Monogioudis: You see the location, you see both of them for 100% of the region, but you see Bokina, what happened. So, I mean, do you see anything on that?
Pantelis Monogioudis: Ha!
Pantelis Monogioudis: It doesn't really make us…
Pantelis Monogioudis: I went to the United States.
Pantelis Monogioudis: It's a loan to the area of her.
Pantelis Monogioudis: They will give me a package, they will kiss me.
Pantelis Monogioudis: D.
Pantelis Monogioudis: You need to look at that.
Pantelis Monogioudis: That's crazy.
Pantelis Monogioudis: That's another… What do you want to set up by the president?
Pantelis Monogioudis: Saying on how we thought we wanted our food truck.
Pantelis Monogioudis: a gusta.
Pantelis Monogioudis: nudos.
Pantelis Monogioudis: You bet.
Pantelis Monogioudis: Wait, should we forget or not?
Pantelis Monogioudis: To avail resident HR.
Pantelis Monogioudis: Meg, while we go home on vacation, so…
Pantelis Monogioudis: That's your quote.
Pantelis Monogioudis: Hey, Admission, how are you? I really wanted to advance on something, but I was applying to apologize, and we make some confusion whether we should apply to it.
Pantelis Monogioudis: Thank you, idi.
Pantelis Monogioudis: And the thing is, the project part is biased, it's not about too much. Like, he's choosing Compute Engineering. Obviously, Compute Engine can see him using smart, like, better than you wish.
Pantelis Monogioudis: But I think you mentioned that, too, you know, like, in…
Pantelis Monogioudis: Computer Engineering CS. So, what is your… what is your advice on, like, EDing to CS or Lumish? He is not very sure on if he wants to teach directly on what is…
Pantelis Monogioudis: or proper CS is connected with the index?
Pantelis Monogioudis: TMUs.
Pantelis Monogioudis: Higher that than you wish.
Pantelis Monogioudis: Rather than just blank.
Pantelis Monogioudis: Eating.
Pantelis Monogioudis: But he's confused by Richard Legends.
Pantelis Monogioudis: Turkey, you know…
Pantelis Monogioudis: Waterlings.
Pantelis Monogioudis: Each tutorial, what is the email should get into.
Pantelis Monogioudis: Beautiful
Pantelis Monogioudis: This is not.