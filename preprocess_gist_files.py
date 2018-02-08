with open("../tmp/filenames.txt", 'r') as f:
	with open("../tmp/gist.txt", 'r') as g:
		files = f.readlines();
		gists = g.readlines();
		with open("../gists.txt", 'w') as s:
			for i in range(0, len(files)):
				s.write(files[i].strip().split('/')[-1]+":"+str(gists[i]))