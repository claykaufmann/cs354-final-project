from lxml import html
import requests

URL = "https://www.vgmusic.com/music/console/nintendo/gba/"


page = requests.get(URL)
webpage = html.fromstring(page.content)

links = webpage.xpath('//a/@href')
midis = list(filter(lambda link: link.endswith(".mid"), links))

for i, mid in enumerate(midis):
    print(i, "/", len(midis))
    r = requests.get(URL + "/" + mid, allow_redirects=True)
    open("songs/" + str(i) + ".mid", 'wb').write(r.content)




