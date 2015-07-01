# converts temp.txt into format for dictionary
with open(temp.txt, "r") as f:
   		content = f.read().splitlines()
f.close()
f = open('champions.txt', 'w')
for item in content:
	f.write(content.replace('\t', ' '))
f.close()	