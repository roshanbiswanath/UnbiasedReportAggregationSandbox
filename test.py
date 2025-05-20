import chromadb, json

client = chromadb.PersistentClient(path="./test_vectors")

d1 = open("demoNews/pahalgam_alzajeera.txt", "r", encoding="utf-8").read()
d2 = open("demoNews/pahalgam_opindia.txt", "r", encoding="utf-8" ).read()
d3 = open("demoNews/ram_mandir_alzajeera.txt", "r", encoding="utf-8").read()
d4 = open("demoNews/ram_mandir_opindia.txt", "r", encoding="utf-8").read()

# collection = client.create_collection(name="test_collection")
collection = client.get_or_create_collection(name="test_collection")
collection.add(
    documents=[d1, d2, d3, d4],
    ids=["1", "2", "3", "4"],
)

print("Documents added to the collection.")

l = [d1,d2,d3,d4]

for i in range(len(l)):
    results = collection.query(
        query_texts=[l[i]], # Chroma will embed this for you
        n_results=4 # how many results to return
    )
    f = open("results"+str(i+1)+".json", "w", encoding="utf-8")
    f.write(json.dumps(results, indent=4, ensure_ascii=False))
    f.close()

results = collection.query(
    query_texts=["""Who: Liverpool vs Arsenal
What: English Premier League
Where: Anfield, Liverpool, United Kingdom
When: Sunday at 4:30pm local time (15:30 GMT)/nFollow Al Jazeera Sport‘s live text and photo commentary stream./nLiverpool were crowned Premier League champions in front of their fans two weeks ago when Arne Slot’s side thrashed Tottenham Hotspur 5-1./nNow they face an Arsenal side wounded by not only their third consecutive second-place finish in the English top-flight but also their elimination from the Champions League on Tuesday./nManager Mikel Arteta has come out fighting with regards to what the Gunners need to do to take the next step in their hunt for silverware – and that starts with their visit to face the Reds./nArteta has urged Arsenal to use the frustration of having to give champions Liverpool a guard of honour on Sunday as fuel to win the Premier League title next season after admitting they have gone “backwards” this term./n“Something has to drive you, motivate you, and pain for this is a good one to use when you really want to do something. It’s the right thing to do, usually as a motivation for next season,” Arteta said of Arsenal’s guard of honour for the champions./n“They’ve been the best team, they’ve been the most consistent, and what Slot and the coaching staff have done has been fascinating, it’s been really good./n“They fully deserve it, and that’s the sport. If somebody is better, you have to accept it and try to reach that level.”/nThe Gunners were beaten 2-1 by Paris Saint-Germain on Wednesday to end their bid to win the tournament for the first time./nIt was a painful loss for Arsenal, who created a host of chances in the early stages of the second leg but could not find a way past inspired PSG keeper Gianluigi Donnarumma./nAfter finishing as runners-up to Manchester City for the previous two seasons, the north Londoners remain without a title since 2004./nLiverpool are in cruise control in the Premier League with a 15-point gap to Arsenal in second./nThe Gunners themselves are only six points from slipping out of the Champions League qualification positions./nMathematically, the Gunners could finish as low as seventh, which would also mean they finish outside the Europa League qualification spots./n“Both teams look forward to playing this game. Difficult to predict,” Reds boss Slot said when asked what kind of game he expects./n“There is a little bit at stake for Arsenal, as I presume they would rather finish second than third or fourth. Difficult to predict if it will be edgy, but it is a game to look forward to.”/n/nThe sides shared a 2-2 draw at Emirates Stadium in October./nBukayo Saka and Mikel Merino twice gave the Gunners the lead, with Virgil van Dijk and Mohamed Salah coming up with the equalisers./nArsenal’s failure to sign a striker in the January transfer window was a big blow in a season marred by long injury absences for Kai Havertz, Gabriel Jesus and Bukayo Saka at various stages./n“In January, it was clear or not? I made a very clear statement, and the statement continues the same. I want the best team, the best players. If we have three goal scorers over 25, bring them in, we’re going to be a much better team, yes,” Arteta said./n“We are there, we are providing the numbers that win you titles. We have to be a little bit luckier, but still do better to make sure that nobody has a season better than you.”/nArsenal have not won at Anfield since 2012 – a match that Arteta played in for the Gunners./nThe north Londoners are, however, on a five-match unbeaten run against the Reds, winning two of those./nRight-back Conor Bradley will start the match, as revealed by Slot in his pre-match news conference. Trent Alexander-Arnold, who has announced he will leave the club this summer, is set to be named among the subs as a result./nJoe Gomez is the only other major absentee for the Reds./nGabriel Jesus, Gabriel Magalhaes, Takehiro Tomiyasu and Kai Havertz all remain sidelined for the Gunners./nHowever, midfielder Jorginho has returned to the match day squad following an injury."""], # Chroma will embed this for you
    n_results=4 # how many results to return
)

f = open("results5.json", "w", encoding="utf-8")
f.write(json.dumps(results, indent=4, ensure_ascii=False))
f.close()