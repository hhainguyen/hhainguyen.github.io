---
title: About
layout: page
---
![Me]({{ site.picture }})

I am a data scientist working at the Consumer Data Research Centre, University of Liverpool.

I am curious in almost everything, especially things that I haven't understood (well, yet).

I have been working on various projects, all related to (small or big) data. Some are about Linked Data (and the Semantic Web if it still exists) to promote tourism and preserve cultural heritage in rural Scottish islands. Some are about who buying what and when with Click & Collect service analysis of a big high-street retailer in the UK. My recent project is about names. Your names can tell many things about you such whether you like pizza or noodles, when will you take holidays, what is your clothing size and styles etc. Sound interesting?

My hobbies, beside data stuffs, is sports and travelling. I am not playing any sports but I am interested in watching them, look at the odds and probability, the geometry of formation etc... I am also keen on places, their cultures and histories. Being a data scientist also helps me to understand things better, especially the things I am passionate about.


## Skills

#### I am interested in almost any programming language. Whatever code that runs (or might run). Below are some of the most favourite
1. **Python** Kinda like a wife. Day to day tasks. Whatever you got stuck, you can go back to this anytime.
2. **R** Kinda like an affair. I've been only doing R since the last few years when I jumped to a data science job, but very efficient for prototyping and visualising (thanks to [Hackley Wickham](https://en.wikipedia.org/wiki/Hadley_Wickham) for his magical *ggplot*)
3. **Java** Kind of the first love. I learnt Java first during high school and uni and it used to be my comfort zone, but since moving to data science, it became too verbal to me. However, with Java being your mother tongue has an advantage in data science. You can write **scala** code without actually knowing Scala... (for that big Spark job :D )
4. **Javascript and the like (HTML, JSON etc)** Like a good friend. Sometimes you cannot use any of the above and have to go back to your old friend. Best for interactive visualisation (well, people are still using Shiny in R or Bokeh in Python or Plot.ly in both but I prefer native things).
5. **Pop11 and Prolog???** Used in my PhD project but haven't got a chance to touch them for the last 5 years.

### Languages without tools are just like talking without doing... Here are the list of my favourite tools
1. **sklearn**, **Numpy**, **Pandas** has almost everything you need in Python for data science tasks. Most of the implementations utilise multi-cores making your life faster. Systematic testing can be done easily using Pipeline...
2. For R there are so many packages thanks to the magical CRAN but the one I mostly used are **data.table**, **caret**, **ggplot2**, **dplyr** (not really, as if you know **data.table** already you wouldn't need to go to dplyr, I just list it here FYI)
3. **D3.js** for all kinds of Javascript-based visualisation and **Leaflet** for maps
4. For **XGBOOST** fans, xgboost in R and Python are really good packages.
5. **Hadoop and Spark** for jobs that deal with large datasets but mainly for preprocessing. My approach would be preprocessing these data using Spark on our clusters and then aggregated them before doing actual analysis on a smaller machine (with GPU of course :D ).


## Projects
#### Below are the list of my projects, sorted in the latest-first order.
+ **GPS traces**: this project looks at peope movement (tracked by mobile devices). We are interested in where people are most interested in, which means of transports they used, and how often.
+ **[Northern Regional Data Facility (NRDF)](http://nrdf.cdrc.ac.uk)** We collected data from many regions in England in different aspects such as Housing, Health, Broadband, Income etc, prepare them for users to download and visualise them on a dashboard for comparison. In this project I mainly used D3.js to visualise the datasets, but behind the scenes is R in which the datasets are cleaned and combined.
+ **Names** is one of my most favourite projects. We look at how to represent information for a name and how such representation can be used to gain more understanding of a user like what is the country of origin, ethnicity, or gender... There will be a few blogs on this topic soon.
+ **Click & Collect** is a project about profiling the customers who used C&C service such as where they are from, what kind of Internet users they are (using the *Internet User Classification* -- this classification is my colleagues' work but it is very interesting, will have a blog on this too)
+ ~~**CURIOUS**~~ **CURIOS** (yeah, a curios project...) is a project about preserve cultural heritage archives such as the rural islands history in a reusable yet linkable formart such as OWL and RDF so that such data can be exploited for tourism promotion. This is an interesting project I had a chance to work on during my time in *dot.rural Digital Economy Hub* (University of Aberdeen).
+ My PhD project was about the use of a traditional AI technique called *Truth Maintenance Systems* to find logical (semantic) error in large databases. I was working on this topic during my study at the CS school, Uni of Nottingham.
