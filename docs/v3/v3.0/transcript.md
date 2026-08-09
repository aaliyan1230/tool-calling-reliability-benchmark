Monika Jotautaite: So, super excited to be here. Thanks for having me. Um, yeah, it's been uh uh I think this this um presentation will be like a little bit all over the place uh because there just like so many things going on currently and I'm happy to talk about all the different aspects of like things I've done and like my current work. Um but just for some background um uh my my journey sort of into AI safety um started quite like early on. Um I guess I since I don't know like uh reading super intelligence and getting I guess I also started a bit within like effective altruism. Um I knew about AI safety or of AI safety for quite a while and like was reading kind of following less wrong reading a bunch of books. Um and then in the meantime I was doing my master and AI masters um I then even when I finished the masters in 2021 it was still kind of like unclear what to do on how to contribute to AI safety.


00:07:06

Monika Jotautaite: So I um continued as a data scientist for two years at this sort of fintech startup. Um and then I think a little bit after Chad GPT came out uh Meter came out with Eval's agenda uh and started creating evaluations and I was like okay I see an opportunity now I I see like kind of a threat model of how to come in and have some impact. So for the first year I think it was like 2024 um I started with by doing a Cena fellowship. So as I I understand this is like quite similar to Prism um in a sense that it's uh specifically geared towards um women woman research projects and it was just great. It was remote but it had a week uh in person and I think it was just like a really great jump start into kind of like networking and community and uh learning about like current research and all the different um sort of threat models and orgs and who's working on what. And then I continued into a pivotal fellowship during which um I also started women in AI safety which I still uh still run um which are sort of meetings in London um for women to network and I also was doing ML for good which is this like AI safety boot camp and I did it as like both a student and a teacher and then after pivotal uh I did a lot of um work with Arcadia Impact.


00:08:32

Monika Jotautaite: So at the time again this was uh a world a world of um where sort of uh agentic evaluations uh I mean as they are now but they were just kind of like um uh being developed a lot of methodologies um were being developed um so I worked a lot on porting different evaluations into inspectai um varcadia and then also on uh UK The AC bounty. So UK AC at the time had this like giant bounty where you can choose a threat model and design a bunch of evals for that. Um and um yeah that was like a big a big period of my time where we were just like training a lot of evals and implementing them. Um and then from that point onwards I started kind of like taking a step back. I guess I I did like about a year of you know working on uh evaluations and I wanted to pivot a little bit by actually I guess I always saw evaluations as a detect detection mechanism. Um and one thing I wanted to do is to like start thinking about solutions and how to kind of like create end goal impact.


00:09:43

Monika Jotautaite: Um and one thing I started thinking about is like what if we start monitoring a you know all of these agents with other agents or other models. Um and at the time I was uh still uh mentored uh by this uh really cool researcher Mary Fuang uh and she sent me AI control paper uh which I read and I was like this is exactly what I've been thinking about. So then I started doing independent research on AI control which ended up by me getting uh a research grant. I found um a researcher to team up with and we um initially wrote a research proposal on uh control scaling trends or like trying to find some sort of scaling laws in how um red team versus blue team interacts between each other. um which at the time was like I think it's like I think still a valid project. It was just a bit too early in where the field was at. Um and then it turned into and specifically the the gap that was missing was just like we don't have enough data to start drawing trends from.


00:10:52

Monika Jotautaite: Um and a way to get data was by creating a bunch of red teaming data, a bunch of like attacks. Um and I guess because I also did like a bunch of eval beforehands, I kind of knew how to do this already. um which turned into a monitoring bench which um is essentially our research project of a lot of red teaming pipelines for monitors that also has this um evaluation data set as as an artifact. Um and that project really uh went really well. I guess like we kind of met the time of the time um the right time like place in time where uh this spring a lot of labs uh released their moni like for the first time ever released their um sort of control style monitors. So for uh Anthropic that is auto mode for OpenAI it's auto review also they released Mythos um and then did some sort of um uh internal testing of their monitoring systems also using monitoring bench which was really cool. So, we got like um some cool citations from this work. Um, and then a lot of opportunities kind of like came uh after this.


00:12:04

Monika Jotautaite: Um, because I think there's just like a huge demand right now in like uh, you know, learning how to figure out monitoring and AI control. Um, so then I chose to kind of continue this uh, at Apollo Research. Uh, we're starting a new team now. Uh, and I'm there uh, as one of their monitoring researchers. Um and at Apollo Research there's kind of like two streams of work currently. Um or maybe three streams of work. So one is um the the main one is developing watcher which is Apollo's um AI safety product for monitoring. Um so for example if you have uh if you're running any sort of like Claude code or Codex or any sort of like a agentic sessions um you have um a way to both have um sort of stepwise monitors and like these a synchronous um bird's eye view of like what's what's going on across all your sessions um and mapping it to threat models um and then so watcher is like the main product. We then do a lot of like monitoring researcher research for watcher and then at the same time we are trying so the the broad goal is to like raise the bar for monitoring across the industry.


00:13:22

Monika Jotautaite: Uh so we also do red teaming um of frontier lab monitors um so that um yeah and I'll talk about each of these um uh in a bit more detail. Um, cool. I might um it might be a bit of a speed run to to have uh leave time for questions, but um yeah, more broadly for monitoring bench um as I said, the main the main takeaway from the project was to um experiment with a lot of different types of red teaming pipelines. Um, and we just did essentially like at the time red teaming was a a pretty like red teaming for AI control specifically to red team different monitors and protocols. Um, it was pretty much like a non-existent field. So, we took a bunch of like really obvious uh strategies to develop good attacks uh which was you know let's try to have human in the loop and let's have humans develop those strategies. Um, another one was okay, let's have um let's automate this now with LLMs like let's have LLMs write a bunch of strategies then let's see okay uh if you have a strategy can the agent execute it how well can it execute it um and then once you have these attack transcripts um you again let it pass through an LLM uh through like different stages and you refine it and one thing that like we learned um is that these Refinement pipelines are just kind of the best thing you and do for attacks because once you already have a


00:15:01

Monika Jotautaite: transcript already have attack inserted into it, refining it then just makes um makes it almost always like even better or what we showed is that refining it makes across all the different sort of transcripts uh better makes makes the text better or like harder for monitors to catch. um which is really cool and I think the main kind of intuition behind it was that uh current uh models are really good at uh they're they like struggle with um coming up with attack, executing attack and like actually making it good and sneaky um like in inside the episode. But if you can like in a way decompose these skills and try to optimize the agent to do each one of those well. So you know come up with a strategy execute strategy uh refine make it like more evasive um then models can do like already really well um which then becomes a good proxy of attacks from future um future actually like potentially misaligned uh uh sneaky models. Uh cool. So this is um I'm not going to like go too much into detail, but this just shows that um the refined strategies um which are the yellow ones um are uh the yellow and blue ones I think.


00:16:25

Monika Jotautaite: Yeah. Um are always like u improve um improve the transcript regardless of what the attack source is. Um cool. Um so that was the monitoring bench. That was essentially uh me figuring out a bunch of um research proposal sending um to uh coicient giving getting funding and executing this research project. Um and then um the next thing uh that I started the like two three months ago now is uh joining Apollo research for their new monitoring team. um and Apollo Apollo research um agenda uh or like what they work on is um kind of all encompassed by by this blog post we have on scalable monitoring agenda. So scalable monitoring agenda is proposing that monitoring as a problem is just extremely tractable and there's three reasons for it. One is that you can create near unlimited data. So we might have very few instances or like increasingly more but still very few instances of like actually misaligned good attacks in the wild. However, we can uh synthetically generate a bunch of this red teaming attack data by taking existing transcripts and inserting those attacks.


00:17:58

Monika Jotautaite: attacks by using control settings um and so on and so forth. So we can actually synthetically create a bunch of uh red teaming sneaky misaligned attack transcripts. They might be somewhat out of distribution from like what uh you know uh in the future models might actually look like but then we can still kind of push those in distributions closer to each other by just um create creating synthetic attacks and then seeing how models actually bea behave in the wild and then you know comparing those. Um we also have this blue team red team framing where you have this unlimited data. You then throw the both benign uh benign data and attack data into monitors. See how monitors do fix monitor problems um as a response like improve monitors and then for the improved monitors you can again use that monitoring signal to um create a bunch of really good attacks. And you can um with this adversarial uh kind of uh training between blue and red team, you can then have this number go up structure which is great for any sort of like machine learning.


00:19:12

Monika Jotautaite: Um and just keep iterate. you you improve blue team, then you improve red teama as a response and then you keep doing this uh in a scalable way and hopefully you can like push the push your monitors to just be you know uh extremely extremely strong against whatever the current capabilities are. Uh there's a bunch of caveats to that um but but broadly this is the the intention. So um because of this Apollo came up with uh watcher as a actual safety product uh where you have uh coding agents um like cloud codeex etc and then watcher goes into sort of two two branches. One is um watcher live where your monitor looks at each um individual coding agent step and approves or blocks it. Um and there's also what your analyze which uh then collects all of the you know session data uh and monitor responses across all the all like workflows uh potentially if if this is like an enterprise it could be across all the different personas. So you know uh different researchers or different security people or uh sort of ops people in a company might have different threat models around them.


00:20:33

Monika Jotautaite: Um and then you can uh as a security person at an org see this bird's eye view of like uh all the coding trajectories that um are happening within your org which is especially good for already the what we see where uh organizations are and individual researchers are just working not just with like a single coding trajectory but like agents are sp spawning out uttering. a agents and so on. Uh so you actually like it really matters that you have a good understanding of what's what's going on. Um, and then a second reason is that I don't know if you've seen u within like the past month, but we had a bunch of uh already agents breaking out of their evaluations um evaluation u sort of sandboxes and evaluation um infrastructure and actually hack hacking real companies. So hopefully um people become like more safety- minded and uh and you know I think everyone's like there's a just a huge huge demand for monitoring for this reason. Um and just to
J Heitzig: Can I ask I wonder what the privacy implications of


00:21:41

Monika Jotautaite: yeah
J Heitzig: of this are in this sense like monitoring everyone's work
Monika Jotautaite: uh what do you mean like privacy or
J Heitzig: kind of a central security officer can kind of has an overview of what everyone is using coding agents for in the arc.
Monika Jotautaite: Uh yeah, I think like you if if you are at an org, you already have um like all orgs already have uh access to coding agent coding agent transcripts. The thing that monitoring adds is just like for specific threat models. If your agent is for example excfiltrating data, if it is like installing you know uh dangerous packages or whatever then like you will get
J Heitzig: I'm not sure. In Germany, I think it's uh in Germany,
Monika Jotautaite: Yeah.
J Heitzig: it's unlawful to uh look at what everyone is using their coding agents for in detail. I think it wouldn't work in Germany. Uh it can't be done.
Monika Jotautaite: Oh, interesting.
J Heitzig: it um that there seems to be problem with this
Monika Jotautaite: I'm very surprised to hear that. Sorry.


00:22:52

J Heitzig: because it's monitoring kind of it's closely monitoring uh workers which is not allowed in Germany
Monika Jotautaite: interesting. Okay. I I I don't know the specifics of sort of German privacy laws, but um
J Heitzig: sounds like the that that for example my employees council would would raise a
Monika Jotautaite: uh
J Heitzig: scandal over if if my institution was trying to implement this, there would be a big uh uh uh um a big push back by the the employees council and rightfully so probably under under German
Monika Jotautaite: I think yeah I mean I think privacy and safety there's always
J Heitzig: law.
Monika Jotautaite: like friction there um because yeah you can either monitor your agents um and protect the fact that like those agents might be like spun off with a single user task and you know we'll do uh will have like a bunch of internal access. They might have access to your uh to your infrastructure, databases, internal documents. whatever. Um, and then you know you're trusting all of that access with uh like agent with all of that access.


00:24:07

Monika Jotautaite: Um, and then yes, in some in some some sense it's like monitoring workers because you have a user task at the top. Um, in some sense it's like literally monitoring the agent of like what it's doing at each step. So yeah, I mean I'm like uh I think in terms of like policy implications, uh it's probably going to be, you know, like a debate on um and I don't really have like sort of strong um uh like particularly strong takes on how this should be approached. I think stuff that I worry about uh in particular is just like the usual control threat models of like uh rogue internal deployments um where labs are testing. So you know at usual enterprises uh you would hopefully use already deployed models that are um you know safety tested and and all of that and still there's risk uh in response to it. Um at inside of labs you also have models that are just being created that have frontier capability and are not safety tested yet. Um and potentially there is like pretty high risk that they are misaligned.


00:25:23

Monika Jotautaite: Um so so sorry just to so this is I think the last slide um just to tie into this like another so we uh are really worried about the wrong internal deployments and then for that we also uh you know work to help Frontier Labs harden their internal monitors internal and external um uh so as an example we recently finished red teaming auto mode which is Anthropic's uh uh sort of stepwise monitor that um they have access both internally and externally. Um uh and uh if you use Claude you can use autoload vivid as well. Um and yeah essentially we like red teamed and provided a bunch of recommendations for that uh Anthropic uh that they also implemented which is really cool. Um and and the goal here is just to make sure that you know you we raise the bar the bar and also like raise the upper bound and also like kind of race to the top for uh for monitoring by also having um our own internal monitors that we can offer to uh enterprises to individuals but then also you know help labs uh improve their monitoring as well.


00:26:36

Monika Jotautaite: Um yeah and that's my work. Okay, I left quite little time for questions, but I can like extend it for like um five 10 minutes. Uh but yeah, uh overall I'm very excited to be here um and and excited to like hear from you. Um, and I don't have too much time to uh answer questions, but I'm always happy to um if you guys just like reach out with your questions over email uh and uh my email is just monica.ai.
Ayesha Imran: Yeah. Uh, thanks a lot. That was Yeah.
Monika Jotautaite: Thanks
Ayesha Imran: Um, that was great. Hillary, if you want if you have a question, you can go
Hilary Torn: Yeah. Yeah, I I do. Thanks, Monica. That was really um great.
Ayesha Imran: ahead.
Hilary Torn: I would love um advice since you first kind of got started by applying for a grant and working on your project. Like, how far along was your project when you applied for the grant? Like, do you you said you found someone?


00:27:35

Hilary Torn: Like, did you apply individually or did you apply with a person? Like any advice for people coming in who think that maybe like starting out applying for a grant and working on a project for 6 to 12 months.
Monika Jotautaite: Yeah.
Hilary Torn: Way to go.
Monika Jotautaite: Yeah. Maybe I should like write a blog post or on this or something cuz I I learned so much. I think it's like a few things that I think are important especially in in today's age. I think even even more so. Um because so the way so I can mostly talk about CG because like that's what I experienced. Um so the best thing you can do I think I think maybe if I did this again maybe that's the what I would optimize for more um is go to um like a fellowship that has extens extensions. So when I started for example pivotal I think for the first time ever maybe they did like an extra bit of funding for uh research extensions now across the board every fellowship does like two months to start and then you can extend to like six or nine or I think for maths maybe it's like almost unlimited like I don't know exactly how it works but like I know that maths has like crazy long extensions where you can just like do research up to the point where you get hired


00:28:57

Monika Jotautaite: Um, I think doing it through CG is a bit of a it's like it has a lot of pros and cons. So, one of the things that one of the pros is that um you can really customize it for yourself. Um, one of the cons is that it is extremely uncertain until the last minute. It can take pretty long time. I think from the moment I sent a bunch of expressions of interest to the moment which I which was maybe like let's say April um then it took like a month or so to hear like um to hear back from them uh with with the next kind of stage um uh for like second stage of applications cuz they do like a bunch of expressions of interest then they do like a full proposal um and then from full proposal I got like an answer that I'll have the or sort of like broad excitement that I'll get the get a grant which took about another like two or 3 months. Um, and when you get like their excitement, it's stills until you like sign a contract which takes like another one or two months, uh, you're still like kind of waiting for funding.


00:30:12

Monika Jotautaite: Um, so it's like, you know, it's like a roller coaster. Um and then and then for both me essentially so I did everything by myself in the initial stages and then by the time I got like into second stage for like two or three proposals I started like looking for teammates because in terms of like it being customizable I kind of knew how I want to work. So I knew that I want to have a teammate. I don't want to do this by myself. um I knew the sort of things that I like agendas that I am excited about and what I you know like if it's like your project and your proposal I think I just really I don't know like I need to kind of really believe in the project to to work really hard on it and then another thing I did correctly I think is to find people in the field already working on similar stuff who would be excite really excited about my project. So for me that was um this guy I met through a fellowship Ollie Matthews and another guy from Redwood Tyler Tracy who someone just like recommended me to like reach out to him and then both of them were like okay yeah these are cool like cool ideas we are happy to supervise you and because they were happy


00:31:22

Monika Jotautaite: to supervise me I put them as like uh in the proposal you can put like people you'll you're you will work with so I put myself my um my friend who you know became my teammate and then uh two mentors and I think if you find correct people those people are also the people that uh coefficient given talks to when they like I'm pretty sure that they reach out to Tyler and was like what do you think about this project and like this person and essentially like referrals are super important in the age of AI where anyone can just you know trajipity a proposal and I think yeah getting getting uh people in the field kind of like vouch for is super important and then uh and then yeah I just like worked extremely hard for like six seven months um and then it it ended up really going really well. I think another thing that I did correctly is just kind of really tried to spend time thinking on like what will matter in like six months. Uh so I did that with the proposal and when I got the funding and I started seeing in the first month that this proposal is not fit for where we are at in the field.


00:32:37

Monika Jotautaite: Uh did a huge pivot which you know was like a huge bet and um and then that ended up really working out cuz I was like no I I think this will matter. Um and um and it just like it did matter. But I think the fact that like industry convert to releasing their monitors at the same time as I released monitor evals. that was like uh you know just extremely lucky lucky thing. Um so yeah so a bunch of things just kind of line lined up. It was like yeah um but hopefully I I would say a much easier way is to just apply to maths and get an extension and just keep working. Um also if you are you know kind of at a time when you are maybe self-sufficient enough uh then getting a grant potentially is the best way if you want to like start something of your own because again it's like you can like get get off the ground. So this was like my another plan of like if I I kind of knew what I want to work on already and then it was just like a lucky thing that Apollo matched my vision.


00:33:45

Monika Jotautaite: Uh but like my main my uh my main kind of plan was to just get like even a bigger grant and a bigger grant and just like keep employing people.
Hilary Torn: Nice. Nice. Thank you for sharing.
Ayesha Imran: Yeah, that is really inspiring.
Monika Jotautaite: Yeah.
Ayesha Imran: Um I think I think one of the most challenging things probably is like you know like you're coming up with ideas and you're as a beginner especially like you're trying to find something novel to work reaching out to people and supervisor like yeah just like short meetings or something just help like supervisor work I
Monika Jotautaite: Yeah.
Ayesha Imran: think that is super important.
Monika Jotautaite: Yeah. Yeah. Yeah. I think it's like easy to kind of in retrospect be like, "Oh, yeah. You know, you did all these But like at each step I think I kind of like always joke that I
Ayesha Imran: So,
Monika Jotautaite: think like a year and a half ago I was like in between projects and I I was like pretty much unemployed for like a couple of months and I was like I don't know like I kind of already did I think two different fellowships.


00:34:40

Monika Jotautaite: I did Arcadia and I was like um I kind of don't want to do kind of like some sort of periodic work or like contract you know that's like six six month six this many months contract or like another fellowship. um I kind of like already wanted to do either something of my own or like find a job but also with jobs it's especially in AI control in London I think only UK AC had a team and by the time I applied they were like oh we already found a person for that position so I was like okay well so I think it took like a lot of kind of diying things yourself and creating opportunities for yourself and just like reaching out to people like reading papers myself and like really understanding them and then uh yeah I don't know
Ayesha Imran: Yeah.
Monika Jotautaite: like I think yeah I think I think like another thing I talked about was just yeah I think I don't remember if it was all Tyler but I was like here's a bunch of ideas I have I will work on AI safety even though I'm like you know unemployed I have like some self-unding for a couple of months uh of these ideas what do you think I should do and then once they kind of say tell you like this is pretty cool.


00:35:55

Monika Jotautaite: You can you can start working on it. get their feedback and just like kind of yeah as you work with people already inside the field you are building connections you're building building your future references you you are building your own kind of like output that you can share um and then I think yeah like all all of these different things I think I did like ported some uh yeah I think I ported like bench as a control setting and it kind of didn't fully make sense to do that for like a bunch of reasons. Um, but like just doing that like tiny project made gave me like a bunch of other ideas how to how to do things. And then I already like met a bunch of then people in control cuz like I think yeah one one thing that like you really want to make use of um and that I think AI safety is just extremely good at is just like pushing people to like connect with each other. Like if you if you meet people and have a bunch of your own ideas and you work on those ideas and you're like, "Here's here's an idea I had. Here's like a finished full project.


00:37:00

Monika Jotautaite: Here's like an idea I had. Here's like a less strong blog post on like me investigating this for a week." I think like if you consistently like come in with with an idea and like crush it even if it's like okay this this project didn't work out for for like these reasons but like you can clearly and sort of correctly explain this to people like it doesn't have to be a positive result for people to like kind of notice and I think people if people see other people being very like ambitious driven hardworking like it'll it'll be noticed for sure. So yeah, I think but but it but it is a strange field in the sense that it's like super competitive and super uncertain and you kind of need to like learn to have your your own taste and
Ayesha Imran: Yes.
Monika Jotautaite: understanding of of these problems and be like okay like what do I think will matter in 6 months and I think for some reason I'm like kind of good at this.
Ayesha Imran: Yeah,
Hilary Torn: Yeah,
Ayesha Imran: I think Yeah,


00:37:57

Hilary Torn: it's a really go.
Ayesha Imran: go
Hilary Torn: Oh, I was just going to say I think the field the field reminds me of a startup itself.
Ayesha Imran: ahead.
Hilary Torn: Like the whole field is one like kind of chaotic startup where we all know what we're doing but no one knows what they're doing at the
Monika Jotautaite: Yeah.
Hilary Torn: same time and we're all like experimenting and failing fast and moving and and shifting and pivoting.
Monika Jotautaite: Yeah.
Hilary Torn: So great.
Monika Jotautaite: Yeah. Yeah.
Hilary Torn: Thank you.
Monika Jotautaite: Yeah. No one knows. No one knows genuinely. Um Yeah. It's like really and it's really tough and it's really like you need to just work so hard. Um, and people that like I don't know uh like people that I look up to, they just like are insane uh at I think having some sort of good takes. Uh but then also just like you know it's like having good predictions about the future or like good project ideas is only like 20% of like it's kind of necessary but not sufficient but like really executing it to like great level that's um and then also like then sharing goes.


00:38:58

Monika Jotautaite: ideas. Like I think one thing that I really appreciate um from like all the Redwood mentorship I got is if they see that your project is good and and that you're kind of like have reasonable takes across you know along the way. I think they like do such a great job at supporting you. Like I was like I kind of um yeah I think this is like fine to share. Like I think uh I still had like a month left of like the main experiments to run. I was like I ran out of budget. Uh I don't know what to do for like my compute budget for the grant. Um and then they're like yeah whatever just here's API keys. Um you know and I think if you if you're like doing good job like people are extremely keen to support you. Um and they're like okay come come give a talk at like control con come you know do this do that like yeah talk to talk to this like researcher at like anthropic talk you know um so I think people are really like extremely extremely supportive from what I found which is like really wonderful cuz like yeah everyone cares


00:40:01

Ayesha Imran: Yeah. Yeah. I think I had a bunch of questions, but we probably answered most of these like about how to I think Yeah. One thing that I would still like to ask like you know you mentioned you can reach out to people and ask you know if they're open to supervising and stuff but like uh what do you think is the best strategy like sometimes you just have ideas sometimes you have like you know you've done an early pilot sometimes you have already a work a research project that you worked on what do you think you know is the best thing like how what is the best way to reach out to uh people and you know ask for feedback and
Monika Jotautaite: Mhm. Um, so it depends if you're like called emailing of or if there's like slightly more kind of uh you already like met them and talked to them or something like that. Um, I think the usual way so if you're if you're just like, "Oh, please mentor me, like I would ignore this immediately." Uh there's um


00:40:58

Monika Jotautaite: like if you are reaching out to someone who's pretty busy like you need to kind of like still offer them something in in some way like so what you can offer is um so one so that's why fellowships are really good because they kind of create all this infrastructure but like let's suppose you're unemployed and you're just like I don't know maybe rejected from all the fellowships because they're like crazy hard to get into um and uh you're just like unemployed reaching out to people. What I would do is like spend some time being like what I think is like the most important to work on. Um come up with okay so again like let's I'll just use like my myself as an example. I was like okay I really believe in AI control now. I want to like upskilling in AI control. I want to like do cool projects. Um uh so I first just read a bunch of papers. The lucky thing is that there's at the time like a year ago there was like not actually very few papers to read.


00:42:00

Monika Jotautaite: So you can like pretty much read all the I don't know five papers. Um then I was like okay what's missing? what I think would be cool to work on. And then I was mentored at the time by Mary Fong and he and she was just like, "Oh, you know, reach out to Tyler Tracy." And then or yeah, someone told me to like reach out to him and then I was like, "Here's okay. Uh, I'm like applying to Open Phil or uh CG um the for this uh independent research stuff. I here's like my three ideas for AI control that I want to work on." you write a paragraph on on each I think because I was also writing those proposals I had like Google docs where they can comment um and then of the paragraph each he was like okay here's my ranking then you take the first one the first that okay this is what they think is like the most important uh write up a Google doc like an actually good Google doc and then just like start sending out to people to to give comments and I guess I was like also with with both Olly and Tyler who ended up being my mentors.


00:43:06

Monika Jotautaite: Um just kind of like okay um this is the project I'll kind of like zoom into because both of you like prefer this. Um and then yeah you just kind of like work like they are already excited about the idea. So if if you have for a researcher something that they're like okay this is a cool project. Um like they'll want to know the progress of it. Um, and then I think I was just like pretty self-driven and like proactive and they're like if you if they're like, "Oh, you should go talk to that person." You immediately email them, get the get the conversation out. And I I was like still, you know, for those two or three months like unemployed, just waiting for open fill answer. But then I just like was doing the project kind of already. Um, and then it and this is like extremely risky. I don't want to like necessarily propose this as a strategy because I I think I was also lucky. Um but yeah, when I got the funding, I already had this, you know, months of kind of preparation and like thinking.


00:44:08

Monika Jotautaite: Um yeah, so that it was just like it just worked out really really well. I think if I if I started again, I would probably still do the same broad things of just like have good takes, you know, be like, "Okay, uh, you're building control arena. Here's um, you know, PR of like me implementing this and that." And if they like notice that you're just turnurning out good work for like nothing, you're just like, "Yeah, somewhere in Lithuania, uh, you know, throwing code at like good code at them." like yeah people people will kind of pay attention. Um, and now being kind of like a bit on the other side, I just also noticed the same thing. Like some people are just, hey, can I have a call with like no background? And then other people are like, here's like my research take, here's like a Google doc with a proposal. And this is a T. TLDDR if you're like interested in the TLDDR you'll read the doc and then of course like I want to help like if it takes you know just a me commenting on on a Google doc or if it takes having a quick call to like steer some project like I'm super happy to do that if I think it's like promising and I think other people are kind of the same and yeah I think maybe another intuition is


00:45:22

Monika Jotautaite: like it's really strange because I feel like there's both a lot of like influx of people uh into the field. But then like for example, we are trying to hire now at Apollo and like still hiring is so hard. Even though there's like all these fellowships and stuff like that, like monitoring just is um so kind of hungry for talent right now. uh or like AI control um because you know you have these autonomous agents just like running around um doing like god knows what and you really want to like know about it and it's still you know there's like so few people upskilled so you if you can like kind of bridge the gap and upskill yourself which I think is just like people struggle with it because people struggle with like self-arning they need to be put in like environments and mentorships etc but if you can kind of like push yourself to to learn things which I think now is like even you know coding projects everything's like so much easier with AI like yeah um I think you're already kind of like ahead of just like create opportunities for yourself and upskill yourself and try to


00:46:35

Ayesha Imran: Yeah, I think at the end of the day, what matters is like what you can completely show and what work that you've put in. I think that's super important. Thanks a lot.
Monika Jotautaite: Yeah. Yeah.
Ayesha Imran: Like this was this was really really useful.
Monika Jotautaite: Nice. Cool. Uh yeah, happy to answer more questions over email, but uh I'll need to run. Um yeah, it was really great to meet you. Best of
Ayesha Imran: Thanks a lot Monica.
