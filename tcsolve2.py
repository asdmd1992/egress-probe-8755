#!/usr/bin/env python3
"""
tcsolve2.py — Tencent Cloud password-recovery chain solver (captcha -> reset -> login).

Chain (account 200044417226 / admin@qqlink.com):
  recover page -> submit email -> TCaptcha popup (#tcaptcha_iframe_dy, click_image_uncheck:
  instruction word + 3x2 grid of 6 Hunyuan-AI images) -> MobileNetV2 ONNX classify tiles ->
  click matching tile -> 确定 -> ticket -> sendRecoverEmail fires -> email with 6-digit code ->
  poll IMAP -> open reset form -> set new password -> "密码修改成功"

Modes:
  solve                     full chain above (captcha + sendRecoverEmail + IMAP + reset)
  reset [url]               open reset URL (with authcode) -> set new password
  login                     email+new password login (handle email-verify step if asked),
                            dump account info + console cookies

Env:
  TC_EMAIL     target email (default admin@qqlink.com)
  TC_NEWPASS   new password (8-20 chars, letter+digit+special)
  TC_MAX       max captcha attempts (default 3)
  TC_IMAP_USER / TC_IMAP_PW   IMAP creds (Gmail app password)
  TC_IMAP_WAIT max seconds to wait for the email (default 180)
  TC_RESET_URL reset url for mode=reset
  TC_MODE      mode if not given as argv
"""
import os, sys, time, json, re, random, io, base64, imaplib, email as eml
import urllib.request, urllib.error
from email.header import decode_header, make_header

import numpy as np
import cv2

EMAIL = os.environ.get("TC_EMAIL", "admin@qqlink.com")
NEWPASS = os.environ.get("TC_NEWPASS", "")
MAX_CHALLENGES = int(os.environ.get("TC_MAX", "3"))
IMAP_USER = os.environ.get("TC_IMAP_USER", EMAIL)
IMAP_PW = os.environ.get("TC_IMAP_PW", "")
IMAP_WAIT = int(os.environ.get("TC_IMAP_WAIT", "180"))
MODE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TC_MODE", "solve")
RESET_URL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TC_RESET_URL", "")
EVID = "evidence"
os.makedirs(EVID, exist_ok=True)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model", "mobilenetv2.onnx")
SYNSET = os.path.join(HERE, "model", "synset.txt")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def log(*a):
    print("[tc2]", *a, flush=True)


def dl(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://cloud.tencent.com/"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def save_json(name, obj):
    with open(os.path.join(EVID, name), "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


# ---------------- browser ----------------
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


def make_ctx(pw):
    browser = pw.chromium.launch(
        headless=True,
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox",
              "--disable-dev-shm-usage", "--lang=zh-CN", "--window-size=1280,900"],
    )
    ctx = browser.new_context(
        viewport={"width": 1280, "height": 860},
        user_agent=UA, locale="zh-CN", timezone_id="Asia/Shanghai",
    )
    ctx.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        window.chrome = window.chrome || {runtime:{}};
        Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
        const origQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (p) => p.name === 'notifications'
            ? Promise.resolve({state: Notification.permission}) : origQuery(p);
    """)
    return browser, ctx


def find_captcha_frame(page):
    for f in page.frames:
        u = f.url
        if "captcha" in u or "tcaptcha" in u or "cap_union" in u or "gtimg" in u:
            return f
    return None


def wait_captcha_frame(page, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = find_captcha_frame(page)
        if f:
            return f
        time.sleep(0.5)
    return None


def shot(page, name):
    page.screenshot(path=os.path.join(EVID, name), full_page=False)


def human_delay(a=0.4, b=1.2):
    time.sleep(random.uniform(a, b))


# ---------------- classification ----------------
def load_labels(path):
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                labels.append(parts[1].split(",")[0].strip())
            else:
                labels.append(parts[0])
    return labels


CATS = {
 'fruit': ['apple','apricot','avocado','banana','blackberry','blueberry','cantaloupe','cherry','citrus','coconut','cranberry','currant','custard apple','date','fig','grape','grapefruit','guava','honeydew','jackfruit','kiwi','lemon','lime','lychee','mango','melon','mulberry','nectarine','orange','papaya','passion fruit','peach','pear','persimmon','pineapple','plum','pomegranate','quince','raisin','raspberry','strawberry','tangerine','watermelon'],
 'vegetable': ['artichoke','asparagus','beet','bell pepper','broccoli','brussels sprout','cabbage','carrot','cauliflower','celery','chili','corn','cucumber','eggplant','endive','garlic','ginger','kohlrabi','leek','lettuce','mushroom','okra','onion','parsnip','pea','potato','pumpkin','radish','rhubarb','rutabaga','shallot','spinach','squash','sweet potato','turnip','zucchini','yam'],
 'food': ['bagel','barbecue','biscuit','bread','burrito','cake','candy','cannelloni','cheese','chocolate','chop suey','club sandwich','consomme','cookie','croissant','dough','doughnut','dumpling','eggs','enchilada','espresso','falafel','fondue','french loaf','french toast','fried','guacamole','hamburger','hotdog','ice cream','icecream','jelly','menu','omelet','oyster','pasta','pastry','pie','pizza','popcorn','pretzel','pudding','sandwich','sauce','sausage','soup','spaghetti','sundae','taco','toast','trifle','waffle','food'],
 'animal': ['dog','hound','terrier','spaniel','retriever','shepherd','collie','mastiff','poodle','schnauzer','pinscher','wolf','coyote','dingo','dhole','fox','husky','malamute','samoyed','chow','pug','boxer','beagle','basset','bloodhound','greyhound','whippet','saluki','dalmatian','rottweiler','doberman','bulldog','pomeranian','pekinese','lhasa','shih','griffon','kelpie','briard','komondor','kuvasz','schipperke','groenendael','malinois','bouvier','appenzeller','entlebucher','affenpinscher','basenji','leonberg','newfoundland','pyrenees','keeshond','cairn','airedale','boston','yorkshire','norfolk','norwich','sealyham','lakeland','bedlington','kerry','scotch','tibetan','silky','wheaten','west highland','bullterrier','borzoi','elkhound','otterhound','deerhound','weimaraner','vizsla','setter','clumber','sussex','cocker','sheepdog','cat','tabby','tiger cat','egyptian cat','lynx','cougar','jaguar','leopard','lion','tiger','cheetah','bear','sloth','otter','skunk','raccoon','badger','pig','hog','boar','cattle','ox','bison','buffalo','ram','ewe','lamb','goat','llama','alpaca','camel','dromedary','gazelle','deer','hartebeest','buck','impala','antelope','gnu','zebra','giraffe','elephant','rhino','hippopotamus','walrus','seal','sea lion','dolphin','porpoise','whale','bat','hedgehog','porcupine','hamster','guinea pig','mouse','rat','squirrel','chipmunk','marmot','beaver','rabbit','hare','kangaroo','wallaby','koala','opossum','platypus','echidna','mole','shrew','weasel','mink','ferret','polecat','mongoose','meerkat','hyena','aardvark','armadillo','tapir','okapi','capybara','chinchilla','gerbil','vole','lemming','pika','panda','monkey','macaque','orangutan','chimpanzee','gorilla','gibbon','baboon','mandrill','proboscis','tarsier','marmoset','lemur','aye-aye','bird','hen','cock','rooster','chicken','duck','goose','swan','turkey','peacock','pheasant','partridge','quail','grouse','pigeon','dove','crow','raven','rook','jackdaw','magpie','jay','chough','starling','blackbird','thrush','robin','sparrow','finch','canary','bunting','warbler','flycatcher','wren','nuthatch','creeper','tit','chickadee','kingfisher','woodpecker','barbet','toucan','hornbill','cuckoo','owl','eagle','hawk','kite','vulture','falcon','kestrel','harrier','osprey','condor','buzzard','kookaburra','parrot','macaw','cockatoo','lorikeet','parakeet','budgerigar','lovebird','hummingbird','swift','swallow','martin','lark','pipit','wagtail','shrike','vireo','oriole','grackle','cowbird','meadowlark','redbird','cardinal','tanager','grosbeak','siskin','goldfinch','indigo','bluebird','waxwing','mockingbird','catbird','thrasher','dipper','kinglet','gnatcatcher','ostrich','emu','rhea','cassowary','kiwi','penguin','albatross','petrel','shearwater','fulmar','gannet','booby','cormorant','shag','anhinga','frigatebird','tropicbird','pelican','heron','egret','bittern','ibis','spoonbill','stork','flamingo','crane','limpkin','rail','crake','gallinule','moorhen','coot','bustard','plover','lapwing','dotterel','sandpiper','curlew','godwit','dowitcher','snipe','woodcock','phalarope','avocet','stilt','oystercatcher','jacana','gull','tern','skimmer','auk','guillemot','murre','puffin','razorbill','dovekie','auklet','murrelet','fish','shark','ray','skate','eel','salmon','trout','bass','perch','pike','catfish','cod','haddock','halibut','flounder','sole','tuna','mackerel','sardine','anchovy','herring','carp','goldfish','koi','guppy','tetra','angelfish','clownfish','seahorse','pufferfish','blowfish','swordfish','marlin','sailfish','sturgeon','paddlefish','gar','bowfin','mudpuppy','newt','salamander','frog','toad','turtle','terrapin','tortoise','snake','lizard','gecko','iguana','chameleon','anole','skink','monitor','komodo','crocodile','alligator','caiman','gavial','worm','snail','slug','octopus','squid','cuttlefish','nautilus','crab','lobster','crayfish','shrimp','prawn','krill','barnacle','spider','scorpion','tick','mite','bee','wasp','hornet','ant','termite','fly','mosquito','gnat','midge','beetle','weevil','ladybug','ladybird','firefly','lightning bug','cockroach','mantis','cricket','grasshopper','locust','cicada','aphid','leafhopper','treehopper','spittlebug','scale insect','mealybug','whitefly','psyllid','thrips','earwig','mayfly','dragonfly','damselfly','stonefly','lacewing','antlion','dobsonfly','fishfly','alderfly','snakefly','scorpionfly','caddisfly','moth','butterfly','skipper','silkworm','caterpillar','centipede','millipede','wood louse','pill bug','sow bug','amphipod','isopod','copepod','ostracod','tadpole','axolotl'],
 'vehicle': ['car','automobile','ambulance','beach wagon','cab','convertible','coupe','cruiser','golfcart','jeep','limousine','minivan','pickup','racer','sports car','stock car','taxi','tow truck','trailer','truck','lorry','van','bus','minibus','trolleybus','train','rail','locomotive','freight','subway','tram','streetcar','cable car','gondola','airliner','airplane','aeroplane','aircraft','warplane','fighter','bomber','jet','biplane','glider','helicopter','airship','blimp','zeppelin','balloon','rocket','spacecraft','shuttle','ship','boat','liner','yacht','sailboat','catamaran','trimaran','kayak','canoe','rowboat','ferry','tugboat','trawler','barge','freighter','tanker','submarine','destroyer','cruiser','battleship','carrier','hovercraft','hydrofoil','jet ski','motorboat','powerboat','snowmobile','motorcycle','scooter','moped','bicycle','bike','tricycle','unicycle','segway','skateboard','roller skate','inline skate','wheelchair','tank','bulldozer','excavator','crane','forklift','tractor','combine','plow','snowplow','cart','wagon','carriage','chariot','rickshaw','sled','sleigh','dog sled','surrey','oxcart','go-kart','quad bike','atv'],
 'furniture': ['chair','sofa','couch','loveseat','seat','bench','stool','throne','table','desk','counter','cabinet','cupboard','dresser','chest','commode','bureau','sideboard','buffet','credenza','hutch','wardrobe','armoire','closet','bookcase','shelf','bookshelf','rack','bed','bunk','hammock','crib','cradle','bassinet','divan','ottoman','footstool','cushion','pillow','mattress','blanket','quilt','comforter','duvet','bedspread','spread','rug','carpet','mat','curtain','drapery','drape','blind','shade','window','door','lamp','floor lamp','table lamp','chandelier','lantern','candle','candelabrum','candlestick','mirror','vase','urn','planter','clock','grandfather','cuckoo clock','alarm clock','watch','television','tv','radio','stereo','speaker','refrigerator','freezer','stove','oven','microwave','toaster','toaster oven','dishwasher','washer','dryer','iron','vacuum','fan','air conditioner','heater','radiator','fireplace','hearth','furniture','furnishings'],
 'electronic': ['phone','cellphone','cellular','smartphone','iphone','telephone','cordless','payphone','handset','computer','laptop','notebook','netbook','desktop','workstation','server','mainframe','supercomputer','tablet','ipad','palmtop','pda','calculator','keyboard','mouse','monitor','screen','display','printer','scanner','copier','fax','projector','camera','camcorder','webcam','video camera','telescope','microscope','binoculars','magnifying glass','speaker','headphone','earphone','earbud','microphone','amplifier','receiver','transmitter','antenna','satellite dish','router','modem','switch','hub','adapter','charger','battery','power bank','usb','flash drive','memory card','hard drive','ssd','optical drive','cd','dvd','blu-ray','record','vinyl','turntable','radio','stereo','boombox','walkman','mp3','ipod','television','tv','led','lcd','plasma','oled','smart tv','remote','game console','playstation','xbox','nintendo','controller','joystick','vr','headset','goggles','smartwatch','fitness tracker','drone','quadcopter','robot','vacuum robot','e-reader','kindle','gps','navigation','radar','sonar','electronic','electric','digital','device','gadget','appliance'],
 'plant': ['plant','flower','blossom','bloom','rose','tulip','daisy','sunflower','lily','lotus','orchid','peony','daffodil','narcissus','crocus','hyacinth','iris','lilac','jasmine','gardenia','magnolia','azalea','rhododendron','camellia','hibiscus','poppy','marigold','zinnia','petunia','begonia','geranium','lavender','violet','pansy','snapdragon','gladiolus','dahlia','chrysanthemum','aster','cosmos','dandelion','clover','buttercup','anemone','bluebell','snowdrop','tree','shrub','bush','bark','trunk','branch','twig','leaf','leaves','foliage','needle','pine','fir','spruce','cedar','oak','maple','birch','willow','poplar','aspen','elm','ash','beech','chestnut','walnut','hickory','sycamore','plane','linden','acacia','eucalyptus','palm','coconut palm','date palm','bamboo','cactus','succulent','fern','moss','lichen','grass','lawn','meadow','weed','herb','sprout','seedling','sapling','grove','forest','woodland','jungle','rainforest','vine','ivy','creeper','climber','honeysuckle','wisteria','bougainvillea','hydrangea','garden','orchard','plantation','botanical','floral','flora','vegetation'],
 'nature': ['mountain','hill','peak','summit','volcano','cliff','canyon','gorge','valley','ridge','glacier','iceberg','snow','snowflake','ice','frost','hail','rain','raindrop','rainbow','storm','thunder','lightning','cloud','sky','sun','sunshine','sunset','sunrise','moon','moonlight','crescent','star','starry','constellation','galaxy','milky way','planet','earth','world','globe','ocean','sea','wave','surf','tide','shore','beach','coast','sand','dune','desert','oasis','river','stream','creek','brook','waterfall','rapids','lake','pond','pool','reservoir','swamp','marsh','bog','wetland','delta','estuary','fjord','peninsula','island','isle','reef','atoll','lagoon','bay','gulf','cape','port','harbor','nature','landscape','scenery','environment','terrain','topography','geology','fossil','mineral','rock','stone','boulder','pebble','gravel','crystal','gem','diamond','gold','silver','metal','iron','copper','bronze','brass','steel','tin','lead','mercury','zinc','nickel','chrome','aluminum','aluminium','alloy','ore','mine','quarry','cave','cavern','grotto','tunnel','crater','geyser','hot spring','mud','clay','silt','soil','earth'],
 'kitchen': ['kitchen','cookware','pot','pan','skillet','frying pan','saucepan','wok','kettle','teapot','coffeepot','percolator','press','french press','moka','dutch oven','casserole','baking dish','roasting pan','sheet pan','muffin tin','cake pan','pie pan','loaf pan','griddle','grill','barbecue','bbq','stove','burner','hob','oven','microwave','toaster','air fryer','slow cooker','crockpot','instant pot','pressure cooker','steamer','rice cooker','blender','juicer','food processor','mixer','stand mixer','hand mixer','immersion blender','chopper','grinder','coffee grinder','mill','mortar','pestle','rolling pin','whisk','spatula','tongs','ladle','scoop','colander','strainer','sieve','sifter','grater','zester','peeler','corer','masher','cutter','knife','chef knife','paring knife','cleaver','butcher','bread knife','carving','steel','sharpener','cutting board','chopping board','plate','dish','bowl','saucer','cup','mug','glass','goblet','tumbler','flute','snifter','stein','tankard','pitcher','jug','carafe','decanter','bottle','flask','thermos','jar','can','tin','container','tupperware','storage','basket','tray','platter','serving','utensil','silverware','flatware','cutlery','fork','spoon','tablespoon','teaspoon','soup spoon','dessert spoon','butter knife','steak knife','chopstick','chopsticks','skewer','toothpick','napkin','tablecloth','placemat','coaster','trivet','potholder','oven mitt','dish towel','sponge','brush','scrubber','soap','detergent','dishwasher','sink','faucet','tap','countertop','counter','cupboard','cabinet','pantry'],
 'stationery': ['pen','pencil','pencil case','crayon','marker','highlighter','eraser','sharpener','ruler','compass','protractor','set square','triangle','notebook','notepad','paper','sheet','card','postcard','envelope','letter','document','file','folder','binder','clip','paperclip','staple','stapler','pin','pushpin','thumbtack','tape','adhesive','glue','stick','paste','scissors','cutter','blade','exacto','hole punch','puncher','laminator','book','textbook','novel','magazine','journal','diary','calendar','planner','agenda','ledger','album','scrapbook','sketchbook','canvas','easel','brush','paintbrush','paint','palette','ink','dye','pigment','watercolor','acrylic','oil paint','chalk','board','whiteboard','blackboard','chalkboard','bulletin','notice','stamp','inkpad','wax seal','ribbon','string','twine','cord','rope','thread','yarn','needle','thimble','sewing','knitting','crochet','embroidery','cross-stitch','stationery','office','school','classroom','study','desk'],
 'tool': ['hammer','mallet','sledgehammer','wrench','spanner','socket','ratchet','screwdriver','phillips','flathead','allen','hex','pliers','pincers','vise','clamp','c-clamp','bar clamp','pipe wrench','monkey wrench','adjustable','cutters','wire cutter','bolt cutter','snips','shears','saw','handsaw','hacksaw','chainsaw','circular saw','table saw','miter saw','jigsaw','scroll saw','band saw','rip saw','crosscut','bow saw','coping saw','fret saw','keyhole saw','drywall saw','hole saw','blade','axe','hatchet','adze','chisel','gouge','plane','spokeshave','scraper','file','rasp','sandpaper','emery','grinder','angle grinder','bench grinder','sander','belt sander','orbital sander','polisher','buffer','drill','drill bit','bore','auger','brace','bit','countersink','tap','die','threader','reamer','broach','router','planer','jointer','lathe','mill','milling','cnc','press','drill press','arbor','spindle','chuck','anvil','forge','furnace','kiln','torch','blowtorch','welder','welding','soldering','solder','iron','glue gun','hot glue','staple gun','nail gun','rivet','riveter','caulk','caulking','putty','spatula','trowel','float','plaster','tape measure','measuring tape','level','spirit level','laser level','square','try square','combination square','framing square','speed square','bevel','caliper','micrometer','gauge','plumb','chalk line','string line','marking','scribe','awl','punch','center punch','nail','screw','bolt','nut','washer','rivet','anchor','sleeve','stud','dowel','pin','key','wedge','shim','spacer','grommet','eyelet','hook','ring','chain','cable','wire','rope','tool','hardware','workshop','garage','shed','workbench'],
 'sport': ['ball','soccer','football','basketball','baseball','softball','cricket','tennis','table tennis','ping-pong','badminton','volleyball','handball','water polo','rugby','american football','gridiron','hockey','ice hockey','field hockey','lacrosse','polo','golf','bowling','skittles','billiards','pool','snooker','carom','darts','archery','shooting','fencing','boxing','wrestling','judo','karate','taekwondo','kung fu','sumo','weightlifting','powerlifting','bodybuilding','gymnastics','trampoline','acrobatics','diving','swimming','surfing','windsurfing','kitesurfing','kayaking','canoeing','rowing','sailing','yachting','skating','ice skating','figure skating','speed skating','roller skating','inline skating','skiing','alpine skiing','cross-country','snowboarding','sledding','bobsled','luge','skeleton','cycling','bmx','mountain biking','road cycling','track cycling','horse racing','equestrian','show jumping','dressage','eventing','athletics','running','sprinting','marathon','hurdles','relay','javelin','discus','hammer throw','shot put','pole vault','high jump','long jump','triple jump','race walking','triathlon','pentathlon','decathlon','sport','game','match','tournament','championship','competition','olympic','stadium','arena','gym','court','field','track','pitch','rink'],
 'clothing': ['clothing','apparel','garment','dress','gown','robe','skirt','blouse','shirt','t-shirt','tshirt','tee','polo','sweater','cardigan','hoodie','jacket','coat','parka','anorak','raincoat','trench','blazer','suit','tuxedo','vest','waistcoat','jean','jeans','trouser','trousers','pants','slacks','shorts','underwear','underpants','briefs','boxers','lingerie','brassiere','bra','panties','stocking','hosiery','sock','socks','shoe','shoes','sneaker','sneakers','boot','boots','sandal','slipper','loafer','pump','heel','clog','moccasin','espadrille','sabot','gumboot','wellington','flip-flop','thong','cleat','spike','glove','mitten','scarf','muffler','tie','bow tie','necktie','hat','cap','beret','sombrero','cowboy hat','top hat','bowler','fedora','panama','straw hat','bonnet','hood','helmet','headgear','mask','veil','wig','turban','headband','bandana','belt','suspenders','apron','shawl','cape','poncho','sari','kimono','bathrobe','nightgown','pajamas','swimsuit','bikini','trunks','wetsuit','uniform','costume','jersey','jumper','overalls','dungarees','coveralls','jumpsuit','leotard','tights','leggings','collar','cuff','pocket','button','zipper','zip','buckle','lace','ribbon','bow','fabric','cloth','textile','yarn','thread','knit','wool','cotton','silk','denim','leather','fur','feather','down','quilted','padded','waterproof','water-repellent'],
}

# Chinese instruction -> coarse category
INST_CAT = {
    "苹果": "fruit", "梨": "fruit", "香蕉": "fruit", "葡萄": "fruit", "草莓": "fruit",
    "樱桃": "fruit", "橙子": "fruit", "西瓜": "fruit", "桃子": "fruit", "菠萝": "fruit",
    "柠檬": "fruit", "猕猴桃": "fruit", "芒果": "fruit", "火龙果": "fruit", "蓝莓": "fruit",
    "山竹": "fruit", "榴莲": "fruit", "石榴": "fruit", "柿子": "fruit", "橘子": "fruit",
    "柚子": "fruit", "李子": "fruit", "杏": "fruit", "椰子": "fruit", "木瓜": "fruit",
    "哈密瓜": "fruit", "甜瓜": "fruit", "杨梅": "fruit", "桑葚": "fruit", "树莓": "fruit",
    "黑莓": "fruit", "蔓越莓": "fruit", "无花果": "fruit", "枣": "fruit", "橄榄": "fruit",
    "鳄梨": "fruit", "牛油果": "fruit", "金桔": "fruit", "青柠": "fruit",
    "兔子": "animal", "猫": "animal", "狗": "animal", "鸟": "animal", "鱼": "animal",
    "蝴蝶": "animal", "蜜蜂": "animal", "蜘蛛": "animal", "蛇": "animal", "老虎": "animal",
    "狮子": "animal", "大象": "animal", "长颈鹿": "animal", "斑马": "animal", "猴子": "animal",
    "企鹅": "animal", "乌龟": "animal", "青蛙": "animal", "熊猫": "animal", "狐狸": "animal",
    "鹿": "animal", "松鼠": "animal", "刺猬": "animal", "老鼠": "animal", "仓鼠": "animal",
    "金鱼": "animal", "海豚": "animal", "鲸鱼": "animal", "鲨鱼": "animal", "螃蟹": "animal",
    "龙虾": "animal", "虾": "animal", "章鱼": "animal", "水母": "animal", "海星": "animal",
    "海马": "animal", "蜗牛": "animal", "蚂蚁": "animal", "蜻蜓": "animal", "瓢虫": "animal",
    "萤火虫": "animal", "螳螂": "animal", "蟑螂": "animal", "毛毛虫": "animal", "公鸡": "animal",
    "母鸡": "animal", "小鸡": "animal", "鸭子": "animal", "鹅": "animal", "天鹅": "animal",
    "孔雀": "animal", "猫头鹰": "animal", "老鹰": "animal", "鹦鹉": "animal", "鸽子": "animal",
    "燕子": "animal", "麻雀": "animal", "啄木鸟": "animal", "乌鸦": "animal", "绵羊": "animal",
    "山羊": "animal", "牛": "animal", "奶牛": "animal", "猪": "animal", "马": "animal",
    "驴": "animal", "骆驼": "animal", "袋鼠": "animal", "考拉": "animal", "北极熊": "animal",
    "棕熊": "animal", "狼": "animal", "河马": "animal", "犀牛": "animal", "鳄鱼": "animal",
    "蜥蜴": "animal", "变色龙": "animal", "蝙蝠": "animal", "海豹": "animal", "海狮": "animal",
    "海龟": "animal", "龙": "animal", "恐龙": "animal", "羊": "animal", "梅花鹿": "animal",
    "斑马": "animal", "柴犬": "animal", "柯基": "animal", "哈士奇": "animal", "金毛": "animal",
    "泰迪": "animal", "拉布拉多": "animal", "吉娃娃": "animal", "布偶猫": "animal", "橘猫": "animal",
    "狸花猫": "animal", "仓鼠": "animal", "熊猫": "animal",
    "汽车": "vehicle", "小汽车": "vehicle", "轿车": "vehicle", "出租车": "vehicle", "公交车": "vehicle",
    "卡车": "vehicle", "货车": "vehicle", "消防车": "vehicle", "警车": "vehicle", "救护车": "vehicle",
    "飞机": "vehicle", "客机": "vehicle", "直升机": "vehicle", "热气球": "vehicle", "轮船": "vehicle",
    "帆船": "vehicle", "游艇": "vehicle", "快艇": "vehicle", "潜艇": "vehicle", "火车": "vehicle",
    "高铁": "vehicle", "地铁": "vehicle", "自行车": "vehicle", "摩托车": "vehicle", "电动车": "vehicle",
    "拖拉机": "vehicle", "挖掘机": "vehicle", "吊车": "vehicle", "推土机": "vehicle", "坦克": "vehicle",
    "赛车": "vehicle", "跑车": "vehicle", "房车": "vehicle", "面包车": "vehicle", "皮卡": "vehicle",
    "缆车": "vehicle", "索道": "vehicle", "火箭": "vehicle", "宇宙飞船": "vehicle", "无人机": "vehicle",
    "椅子": "furniture", "沙发": "furniture", "桌子": "furniture", "床": "furniture", "台灯": "furniture",
    "书柜": "furniture", "衣柜": "furniture", "床头柜": "furniture", "茶几": "furniture", "梳妆台": "furniture",
    "凳子": "furniture", "秋千": "furniture", "摇篮": "furniture", "窗帘": "furniture", "地毯": "furniture",
    "钟": "furniture", "挂钟": "furniture", "落地灯": "furniture", "吊灯": "furniture", "镜子": "furniture",
    "柜子": "furniture", "电视柜": "furniture", "鞋柜": "furniture", "餐桌": "furniture", "办公桌": "furniture",
    "手机": "electronic", "电脑": "electronic", "笔记本电脑": "electronic", "台式电脑": "electronic",
    "电视": "electronic", "相机": "electronic", "耳机": "electronic", "手表": "electronic",
    "鼠标": "electronic", "键盘": "electronic", "平板电脑": "electronic", "充电器": "electronic",
    "音箱": "electronic", "麦克风": "electronic", "摄像头": "electronic", "路由器": "electronic",
    "游戏机": "electronic", "遥控器": "electronic", "打印机": "electronic", "投影仪": "electronic",
    "电子手表": "electronic", "智能手表": "electronic", "机器人": "electronic", "扫地机器人": "electronic",
    "帽子": "clothing", "鞋子": "clothing", "衣服": "clothing", "裤子": "clothing", "裙子": "clothing",
    "手套": "clothing", "围巾": "clothing", "眼镜": "clothing", "包": "clothing", "背包": "clothing",
    "书包": "clothing", "钱包": "clothing", "领带": "clothing", "袜子": "clothing", "短裤": "clothing",
    "衬衫": "clothing", "T恤": "clothing", "外套": "clothing", "毛衣": "clothing", "牛仔裤": "clothing",
    "运动鞋": "clothing", "靴子": "clothing", "凉鞋": "clothing", "拖鞋": "clothing", "皮带": "clothing",
    "项链": "clothing", "戒指": "clothing", "耳环": "clothing", "手链": "clothing", "皇冠": "clothing",
    "发卡": "clothing", "头饰": "clothing", "雨伞": "clothing", "太阳镜": "clothing",
    "杯子": "kitchen", "碗": "kitchen", "盘子": "kitchen", "瓶子": "kitchen", "筷子": "kitchen",
    "勺子": "kitchen", "叉子": "kitchen", "刀": "kitchen", "菜刀": "kitchen", "锅": "kitchen",
    "茶壶": "kitchen", "水壶": "kitchen", "水杯": "kitchen", "咖啡杯": "kitchen", "酒杯": "kitchen",
    "饭盒": "kitchen", "保鲜盒": "kitchen", "砧板": "kitchen", "打蛋器": "kitchen", "榨汁机": "kitchen",
    "咖啡壶": "kitchen", "保温杯": "kitchen", "奶瓶": "kitchen", "花瓶": "kitchen",
    "花": "plant", "花朵": "plant", "玫瑰花": "plant", "郁金香": "plant", "向日葵": "plant",
    "百合": "plant", "莲花": "plant", "兰花": "plant", "牡丹": "plant", "菊花": "plant",
    "树": "plant", "树木": "plant", "松树": "plant", "柳树": "plant", "椰子树": "plant",
    "仙人掌": "plant", "竹子": "plant", "树叶": "plant", "枫叶": "plant", "草": "plant",
    "蘑菇": "plant", "蒲公英": "plant", "荷花": "plant", "梅花": "plant", "桃花": "plant",
    "樱花": "plant", "薰衣草": "plant", "绣球花": "plant", "康乃馨": "plant", "山茶花": "plant",
    "太阳": "nature", "月亮": "nature", "星星": "nature", "云": "nature", "彩虹": "nature",
    "海岸": "nature", "海滩": "nature", "悬崖": "nature", "峡谷": "nature",
    "冰川": "nature", "港湾": "nature", "海港": "nature", "码头": "nature",
    "山": "nature", "雪山": "nature", "火山": "nature", "大海": "nature", "海": "nature",
    "湖泊": "nature", "河流": "nature", "瀑布": "nature", "沙漠": "nature", "森林": "nature",
    "雪花": "nature", "冰块": "nature", "石头": "nature", "岩石": "nature", "冰山": "nature",
    "闪电": "nature", "龙卷风": "nature", "沙滩": "nature", "岛屿": "nature", "草原": "nature",
    "蛋糕": "food", "面包": "food", "汉堡": "food", "披萨": "food", "冰淇淋": "food",
    "糖果": "food", "巧克力": "food", "饼干": "food", "甜甜圈": "food", "薯条": "food",
    "爆米花": "food", "寿司": "food", "面条": "food", "饺子": "food", "包子": "food",
    "馒头": "food", "月饼": "food", "粽子": "food", "汤圆": "food", "热狗": "food",
    "三明治": "food", "沙拉": "food", "奶酪": "food", "鸡蛋": "food", "煎蛋": "food",
    "火腿": "food", "培根": "food", "薯片": "food", "棒棒糖": "food", "棉花糖": "food",
    "华夫饼": "food", "可颂": "food", "马卡龙": "food", "布丁": "food", "果冻": "food",
    "曲奇": "food", "蛋挞": "food", "年糕": "food", "饭团": "food", "拉面": "food",
    "咖啡": "food", "牛奶": "food", "果汁": "food", "啤酒": "food", "红酒": "food",
    "可乐": "food", "奶茶": "food", "茶": "food", "汽水": "food",
    "足球": "sport", "篮球": "sport", "网球": "sport", "羽毛球": "sport", "乒乓球": "sport",
    "排球": "sport", "棒球": "sport", "高尔夫球": "sport", "保龄球": "sport", "台球": "sport",
    "橄榄球": "sport", "冰球": "sport", "板球": "sport", "铅球": "sport", "球": "sport",
    "书": "stationery", "笔": "stationery", "铅笔": "stationery", "钢笔": "stationery", "尺子": "stationery",
    "橡皮": "stationery", "剪刀": "stationery", "胶带": "stationery", "订书机": "stationery", "笔记本": "stationery",
    "纸张": "stationery", "信封": "stationery", "画笔": "stationery", "蜡笔": "stationery", "颜料": "stationery",
    "锤子": "tool", "扳手": "tool", "螺丝刀": "tool", "钳子": "tool", "锯子": "tool",
    "斧头": "tool", "电钻": "tool", "钉子": "tool", "螺丝": "tool", "卷尺": "tool",
    "梯子": "tool", "扫把": "tool", "拖把": "tool", "水桶": "tool", "铲子": "tool",
    "锄头": "tool", "镰刀": "tool", "钥匙": "tool", "锁": "tool", "灯泡": "tool",
    "蜡烛": "tool", "灯笼": "tool", "风筝": "tool", "气球": "tool", "玩具": "tool",
    "洋娃娃": "tool", "积木": "tool", "汽车玩具": "tool", "泰迪熊": "tool", "不倒翁": "tool",
    "鼓": "tool", "吉他": "tool", "钢琴": "tool", "小提琴": "tool", "喇叭": "tool",
    "笛子": "tool", "口琴": "tool", "铃铛": "tool", "沙锤": "tool", "琵琶": "tool",
    "房子": "furniture", "城堡": "furniture", "桥": "furniture", "塔": "furniture", "教堂": "furniture",
    "灯塔": "furniture", "喷泉": "furniture", "雕像": "furniture", "风车": "furniture", "凉亭": "furniture",
    "帐篷": "furniture", "蒙古包": "furniture", "摩天轮": "furniture", "旋转木马": "furniture", "滑梯": "furniture",
    "秋千": "furniture", "跷跷板": "furniture", "公交车": "vehicle",
    "胡萝卜": "vegetable", "土豆": "vegetable", "番茄": "vegetable", "黄瓜": "vegetable",
    "辣椒": "vegetable", "洋葱": "vegetable", "大蒜": "vegetable", "西兰花": "vegetable",
    "白菜": "vegetable", "生菜": "vegetable", "茄子": "vegetable", "南瓜": "vegetable",
    "玉米": "vegetable", "花生": "vegetable", "豆角": "vegetable", "豌豆": "vegetable",
    "芹菜": "vegetable", "菠菜": "vegetable", "韭菜": "vegetable", "姜": "vegetable",
    "萝卜": "vegetable", "红薯": "vegetable", "山药": "vegetable", "莲藕": "vegetable",
    "竹笋": "vegetable", "苦瓜": "vegetable", "丝瓜": "vegetable", "冬瓜": "vegetable",
    "蘑菇": "vegetable", "香菇": "vegetable", "金针菇": "vegetable", "木耳": "vegetable",
    "青椒": "vegetable", "彩椒": "vegetable", "芦笋": "vegetable", "秋葵": "vegetable",
    "花菜": "vegetable", "包菜": "vegetable", "油菜": "vegetable", "空心菜": "vegetable",
}

# specific keywords per instruction (overrides category scoring with weight 2)
INST_SPECIAL = {
    "梨": ["pear", "fig", "granny smith", "banana"],
    "苹果": ["apple", "granny smith"],
    "樱桃": ["cherry"],
    "香蕉": ["banana"],
    "葡萄": ["grape"],
    "草莓": ["strawberry"],
    "橙子": ["orange"],
    "西瓜": ["watermelon"],
    "柠檬": ["lemon"],
    "桃子": ["peach"],
    "菠萝": ["pineapple"],
    "猕猴桃": ["kiwi"],
    "芒果": ["mango"],
    "蓝莓": ["blueberry"],
    "石榴": ["pomegranate"],
    "椰子": ["coconut"],
    "无花果": ["fig"],
    "牛油果": ["avocado"],
    "兔子": ["rabbit", "hare"],
    "猫": ["cat", "tabby", "tiger cat", "egyptian cat"],
    "狗": ["dog", "samoyed", "husky", "chow", "pug", "beagle", "labrador", "retriever", "collie", "shepherd", "poodle", "terrier", "spaniel", "mastiff", "wolf", "fox", "coyote", "dingo"],
    "鸟": ["bird", "robin", "sparrow", "finch", "jay", "crow", "magpie", "swallow", "swift", "lark", "tit", "chickadee", "wren", "starling", "blackbird"],
    "鱼": ["fish", "goldfish", "koi", "tuna", "salmon", "trout", "clownfish", "angelfish", "pufferfish", "seahorse"],
    "蝴蝶": ["butterfly", "skipper", "moth"],
    "蜜蜂": ["bee", "honeybee"],
    "蜘蛛": ["spider", "tarantula", "garden spider"],
    "蛇": ["snake", "viper", "cobra", "python", "boa", "mamba", "rattlesnake"],
    "老虎": ["tiger", "tiger cat"],
    "狮子": ["lion", "lioness"],
    "大象": ["elephant", "african elephant", "indian elephant"],
    "长颈鹿": ["giraffe"],
    "斑马": ["zebra"],
    "猴子": ["monkey", "macaque", "orangutan", "chimpanzee", "gorilla", "gibbon", "baboon", "mandrill", "tarsier", "marmoset", "lemur", "proboscis"],
    "企鹅": ["penguin"],
    "乌龟": ["turtle", "terrapin", "tortoise", "mud turtle"],
    "青蛙": ["frog", "tree frog", "bullfrog", "toad"],
    "熊猫": ["panda", "giant panda"],
    "狐狸": ["fox", "red fox", "kit fox", "arctic fox"],
    "鹿": ["deer", "hartebeest", "buck", "stag", "doe", "fawn", "elk", "moose", "caribou", "reindeer", "gazelle", "impala", "antelope"],
    "松鼠": ["squirrel", "fox squirrel", "chipmunk"],
    "刺猬": ["hedgehog"],
    "老鼠": ["mouse", "rat", "hamster", "guinea pig", "vole", "lemming", "marmot", "beaver"],
    "仓鼠": ["hamster", "mouse", "rat"],
    "金鱼": ["goldfish", "koi"],
    "海豚": ["dolphin", "porpoise"],
    "鲸鱼": ["whale", "killer whale", "humpback"],
    "鲨鱼": ["shark", "tiger shark", "hammerhead", "great white"],
    "螃蟹": ["crab", "king crab", "hermit crab"],
    "龙虾": ["lobster", "crayfish", "spiny lobster"],
    "虾": ["shrimp", "prawn"],
    "章鱼": ["octopus"],
    "水母": ["jellyfish"],
    "海星": ["starfish", "sea star"],
    "蜗牛": ["snail"],
    "蚂蚁": ["ant"],
    "蜻蜓": ["dragonfly", "damselfly"],
    "瓢虫": ["ladybug", "ladybird"],
    "公鸡": ["rooster", "cock"],
    "母鸡": ["hen"],
    "鸭子": ["duck", "mallard"],
    "鹅": ["goose", "swan"],
    "天鹅": ["swan"],
    "孔雀": ["peacock"],
    "猫头鹰": ["owl", "screech owl", "horned owl", "barn owl"],
    "老鹰": ["eagle", "bald eagle", "vulture", "hawk", "kite", "falcon", "condor", "osprey"],
    "鹦鹉": ["parrot", "macaw", "cockatoo", "lorikeet", "parakeet", "budgerigar", "lovebird", "african grey"],
    "鸽子": ["pigeon", "dove"],
    "燕子": ["swallow", "swift", "martin"],
    "麻雀": ["sparrow", "house sparrow", "finch"],
    "啄木鸟": ["woodpecker"],
    "乌鸦": ["crow", "raven", "rook", "jackdaw", "magpie", "jay", "chough"],
    "绵羊": ["sheep", "ram", "ewe", "lamb"],
    "山羊": ["goat"],
    "牛": ["cattle", "ox", "bison", "buffalo", "cow"],
    "奶牛": ["cow", "cattle"],
    "猪": ["pig", "hog", "boar", "sow"],
    "马": ["horse", "pony", "sorrel", "zebra", "colt", "foal", "mare", "stallion"],
    "驴": ["ass", "donkey", "mule"],
    "骆驼": ["camel", "dromedary"],
    "袋鼠": ["kangaroo", "wallaby"],
    "考拉": ["koala"],
    "北极熊": ["polar bear", "ice bear"],
    "棕熊": ["brown bear", "grizzly"],
    "狼": ["wolf", "timber wolf", "white wolf", "red wolf", "coyote", "dingo", "dhole"],
    "河马": ["hippopotamus", "hippo"],
    "犀牛": ["rhinoceros", "rhino"],
    "鳄鱼": ["crocodile", "alligator", "caiman", "gavial"],
    "蜥蜴": ["lizard", "gecko", "iguana", "chameleon", "anole", "skink", "monitor", "komodo"],
    "变色龙": ["chameleon"],
    "蝙蝠": ["bat"],
    "海豹": ["seal", "sea lion", "walrus"],
    "海狮": ["sea lion", "fur seal"],
    "海龟": ["turtle", "terrapin", "loggerhead"],
    "恐龙": ["dinosaur", "stegosaurus", "triceratops", "tyrannosaurus", "brontosaurus"],
    "汽车": ["car", "automobile", "convertible", "coupe", "limousine", "sedan", "beach wagon", "cab", "jeep", "racer", "sports car", "stock car", "taxi", "minivan", "pickup", "ambulance", "golfcart", "tow truck", "trailer truck"],
    "小汽车": ["car", "automobile", "convertible", "coupe", "sedan", "beach wagon"],
    "轿车": ["car", "automobile", "sedan", "limousine"],
    "出租车": ["taxi", "cab"],
    "公交车": ["bus", "minibus", "trolleybus", "school bus"],
    "卡车": ["truck", "lorry", "pickup", "tow truck", "trailer truck", "garbage truck", "fire truck"],
    "货车": ["truck", "lorry", "pickup", "tow truck", "trailer truck"],
    "消防车": ["fire engine", "fire truck"],
    "警车": ["police van", "police car"],
    "救护车": ["ambulance"],
    "飞机": ["airliner", "airplane", "warplane", "wing", "jet", "aircraft"],
    "客机": ["airliner"],
    "直升机": ["helicopter"],
    "热气球": ["balloon", "hot-air balloon"],
    "轮船": ["ship", "liner", "boat", "yawl", "freighter", "tanker", "tugboat"],
    "帆船": ["sailboat", "yawl", "catamaran", "trimaran", "yacht"],
    "游艇": ["yacht", "cruiser"],
    "快艇": ["speedboat", "motorboat", "powerboat"],
    "潜艇": ["submarine"],
    "火车": ["train", "locomotive", "freight car", "passenger car", "bullet train"],
    "高铁": ["train", "bullet train", "locomotive"],
    "地铁": ["subway", "underground", "metro"],
    "自行车": ["bicycle", "bike", "mountain bike", "tricycle", "unicycle"],
    "摩托车": ["motorcycle", "moped", "scooter"],
    "电动车": ["scooter", "moped", "segway"],
    "拖拉机": ["tractor", "plow", "harvester", "combine"],
    "挖掘机": ["excavator", "backhoe", "digger"],
    "吊车": ["crane", "truck crane"],
    "推土机": ["bulldozer"],
    "坦克": ["tank"],
    "赛车": ["racer", "sports car", "race car"],
    "跑车": ["sports car", "racer", "convertible"],
    "房车": ["camper", "motor home", "recreational vehicle", "rv"],
    "面包车": ["minivan", "van"],
    "皮卡": ["pickup"],
    "火箭": ["rocket", "missile", "spacecraft", "space shuttle"],
    "宇宙飞船": ["spacecraft", "space shuttle", "capsule"],
    "无人机": ["drone", "quadcopter", "radio-controlled"],
    "椅子": ["chair", "folding chair", "rocking chair", "throne", "bench", "stool"],
    "沙发": ["sofa", "couch", "loveseat", "studio couch", "divan", "ottoman"],
    "桌子": ["table", "dining table", "desk", "counter", "pedestal"],
    "床": ["bed", "bunk", "hammock", "crib", "cradle", "bassinet"],
    "台灯": ["table lamp", "lamp", "lampshade"],
    "书柜": ["bookcase", "bookshelf", "shelf"],
    "衣柜": ["wardrobe", "armoire", "closet", "cabinet"],
    "床头柜": ["night table", "nightstand", "chest"],
    "茶几": ["coffee table", "table"],
    "梳妆台": ["dresser", "vanity", "bureau"],
    "凳子": ["stool", "bench", "seat"],
    "摇篮": ["cradle", "crib", "bassinet"],
    "窗帘": ["curtain", "drapery", "drape", "window shade"],
    "地毯": ["rug", "carpet", "mat"],
    "钟": ["clock", "wall clock", "grandfather clock"],
    "挂钟": ["clock", "wall clock"],
    "落地灯": ["floor lamp", "floor lamp"],
    "吊灯": ["chandelier", "pendant"],
    "镜子": ["mirror"],
    "柜子": ["cabinet", "cupboard", "chest", "dresser", "wardrobe", "armoire", "locker"],
    "电视柜": ["entertainment center", "cabinet", "console"],
    "鞋柜": ["cabinet", "cupboard", "shoe rack"],
    "餐桌": ["dining table", "table"],
    "办公桌": ["desk", "table"],
    "手机": ["cellphone", "cellular telephone", "smartphone", "iphone", "hand-held computer"],
    "电脑": ["computer", "desktop computer", "laptop", "notebook", "monitor", "screen"],
    "笔记本电脑": ["laptop", "notebook", "computer"],
    "台式电脑": ["desktop computer", "computer", "monitor"],
    "电视": ["television", "tv", "screen", "monitor"],
    "相机": ["camera", "reflex camera", "polaroid", "cinema camera", "camcorder", "webcam"],
    "耳机": ["headphone", "earphone", "earmuff"],
    "手表": ["watch", "digital watch", "wristwatch"],
    "鼠标": ["mouse", "computer mouse"],
    "键盘": ["keyboard", "computer keyboard"],
    "平板电脑": ["tablet", "ipad", "hand-held computer"],
    "充电器": ["charger", "adapter", "power supply"],
    "音箱": ["speaker", "loudspeaker", "woofer", "subwoofer"],
    "麦克风": ["microphone", "mic"],
    "摄像头": ["webcam", "camera", "cctv"],
    "路由器": ["router", "modem", "switch", "wi-fi"],
    "游戏机": ["game console", "playstation", "controller", "joystick", "hand-held computer"],
    "遥控器": ["remote control", "remote"],
    "打印机": ["printer", "laser printer", "inkjet"],
    "投影仪": ["projector"],
    "智能手表": ["watch", "smartwatch", "digital watch"],
    "机器人": ["robot", "android", "humanoid"],
    "扫地机器人": ["vacuum", "robot", "vacuum cleaner"],
    "帽子": ["hat", "cap", "sombrero", "cowboy hat", "top hat", "bowler", "fedora", "panama", "straw hat", "bonnet", "beret", "bonnet", "mortarboard", "academic gown", "kimono"],
    "鞋子": ["shoe", "sneaker", "running shoe", "boot", "sandal", "slipper", "loafer", "clog", "moccasin", "espadrille", "gumboot", "cleat", "hiking boot", "cowboy boot"],
    "衣服": ["clothing", "dress", "shirt", "suit", "gown", "robe", "jersey", "sweater", "coat", "jacket", "blazer", "parka", "cardigan", "hoodie", "tunic", "kimono"],
    "裤子": ["jean", "trouser", "pants", "slacks", "shorts", "dungarees", "overalls", "jumpsuit"],
    "裙子": ["dress", "skirt", "gown", "hoopskirt", "sarong", "sari"],
    "手套": ["glove", "mitten", "boxing glove", "baseball glove"],
    "围巾": ["scarf", "muffler", "shawl"],
    "眼镜": ["sunglasses", "eyeglasses", "spectacles", "glasses", "goggles", "bifocals"],
    "包": ["bag", "backpack", "purse", "handbag", "shoulder bag", "satchel", "suitcase", "briefcase", "duffel"],
    "背包": ["backpack", "knapsack", "rucksack", "pack"],
    "书包": ["backpack", "schoolbag", "satchel"],
    "钱包": ["wallet", "billfold", "purse"],
    "领带": ["tie", "bow tie", "necktie", "bolo tie"],
    "袜子": ["sock", "socks", "stocking", "hosiery"],
    "短裤": ["shorts", "trunks", "briefs"],
    "衬衫": ["shirt", "polo", "tee", "t-shirt", "blouse", "jersey"],
    "T恤": ["t-shirt", "tee", "jersey", "polo"],
    "外套": ["coat", "jacket", "parka", "anorak", "raincoat", "trench", "blazer", "overcoat"],
    "毛衣": ["sweater", "cardigan", "pullover", "jersey"],
    "牛仔裤": ["jean", "jeans", "denim"],
    "运动鞋": ["sneaker", "running shoe", "gym shoe", "tennis shoe"],
    "靴子": ["boot", "boots", "cowboy boot", "hiking boot", "gumboot", "wellington"],
    "凉鞋": ["sandal", "espadrille", "flip-flop"],
    "拖鞋": ["slipper", "flip-flop", "clog"],
    "皮带": ["belt", "waist belt"],
    "项链": ["necklace", "pendant", "chain"],
    "戒指": ["ring", "wedding ring"],
    "耳环": ["earring"],
    "手链": ["bracelet", "bangle"],
    "皇冠": ["crown", "tiara", "coronet"],
    "雨伞": ["umbrella", "parasol"],
    "太阳镜": ["sunglasses", "shades"],
    "杯子": ["cup", "mug", "coffee mug", "goblet", "tumbler", "beer glass", "water glass", "wineglass"],
    "碗": ["bowl", "mixing bowl", "soup bowl", "cereal bowl"],
    "盘子": ["plate", "platter", "dish", "tray", "saucer"],
    "瓶子": ["bottle", "water bottle", "beer bottle", "wine bottle", "jar", "flask", "carafe", "decanter", "thermos"],
    "筷子": ["chopstick", "chopsticks"],
    "勺子": ["spoon", "tablespoon", "teaspoon", "ladle", "scoop", "soup spoon"],
    "叉子": ["fork", "pitchfork"],
    "刀": ["knife", "cleaver", "chef knife", "paring knife", "butcher", "cutter", "blade"],
    "菜刀": ["cleaver", "knife", "butcher"],
    "锅": ["pot", "pan", "skillet", "frying pan", "saucepan", "wok", "casserole", "dutch oven", "cauldron", "kettle"],
    "茶壶": ["teapot", "teakettle"],
    "水壶": ["kettle", "teakettle", "pitcher", "jug", "carafe"],
    "水杯": ["cup", "glass", "tumbler", "mug", "water bottle"],
    "咖啡杯": ["coffee mug", "cup", "mug", "espresso"],
    "酒杯": ["wineglass", "goblet", "tumbler", "flute", "snifter", "beer glass"],
    "饭盒": ["lunch box", "lunchbox", "container", "tupperware"],
    "砧板": ["cutting board", "chopping board"],
    "榨汁机": ["juicer", "blender"],
    "咖啡壶": ["coffeepot", "percolator", "french press", "moka"],
    "保温杯": ["thermos", "vacuum flask", "water bottle"],
    "奶瓶": ["baby bottle", "bottle", "nipple"],
    "花瓶": ["vase", "urn", "flower vase"],
    "花": ["flower", "rose", "tulip", "daisy", "sunflower", "lily", "lotus", "orchid", "peony", "daffodil", "crocus", "hyacinth", "iris", "lilac", "jasmine", "gardenia", "magnolia", "azalea", "camellia", "hibiscus", "poppy", "marigold", "zinnia", "petunia", "begonia", "geranium", "lavender", "violet", "pansy", "snapdragon", "gladiolus", "dahlia", "chrysanthemum", "aster", "cosmos", "dandelion", "clover", "buttercup", "anemone", "bluebell", "snowdrop"],
    "花朵": ["flower", "rose", "tulip", "daisy", "sunflower", "lily", "lotus", "orchid", "peony", "daffodil", "blossom", "bloom"],
    "玫瑰花": ["rose", "flower"],
    "郁金香": ["tulip"],
    "向日葵": ["sunflower"],
    "百合": ["lily", "water lily"],
    "莲花": ["lotus", "water lily"],
    "兰花": ["orchid"],
    "牡丹": ["peony"],
    "菊花": ["chrysanthemum", "daisy", "dahlia"],
    "树": ["tree", "oak", "maple", "birch", "willow", "poplar", "aspen", "elm", "ash", "beech", "chestnut", "walnut", "hickory", "sycamore", "plane", "linden", "acacia", "eucalyptus", "palm", "coconut palm", "date palm", "bamboo", "pine", "fir", "spruce", "cedar", "juniper", "yew", "redwood", "sequoia"],
    "树木": ["tree", "oak", "maple", "birch", "willow", "poplar", "aspen", "elm", "ash", "beech", "chestnut", "walnut", "hickory", "sycamore", "plane", "linden", "acacia", "eucalyptus", "palm", "pine", "fir", "spruce", "cedar", "redwood"],
    "松树": ["pine", "fir", "spruce", "cedar", "conifer"],
    "柳树": ["willow"],
    "椰子树": ["coconut palm", "palm", "date palm"],
    "仙人掌": ["cactus"],
    "竹子": ["bamboo"],
    "树叶": ["leaf", "leaves", "foliage"],
    "枫叶": ["maple", "leaf", "leaves"],
    "草": ["grass", "lawn", "meadow"],
    "蘑菇": ["mushroom", "fungus", "toadstool", "agaric"],
    "蒲公英": ["dandelion"],
    "荷花": ["lotus", "water lily"],
    "梅花": ["plum", "blossom", "flower"],
    "桃花": ["peach", "blossom", "flower"],
    "樱花": ["cherry", "blossom", "flower"],
    "薰衣草": ["lavender"],
    "绣球花": ["hydrangea"],
    "康乃馨": ["carnation", "flower"],
    "山茶花": ["camellia"],
    "太阳": ["sun", "sunrise", "sunset"],
    "月亮": ["moon", "crescent", "half moon"],
    "星星": ["star", "starfish"],
    "云": ["cloud", "cumulus", "stratus"],
    "彩虹": ["rainbow"],
    "山": ["mountain", "alp", "volcano", "hill", "peak", "summit"],
    "雪山": ["mountain", "alp", "snow", "glacier", "iceberg"],
    "火山": ["volcano"],
    "大海": ["sea", "ocean", "wave", "surf"],
    "海岸": ["coast", "shore", "beach", "seaside", "sea", "ocean", "wave", "cliff", "bay", "gulf"],
    "海滩": ["beach", "sand", "seashore", "coast"],
    "悬崖": ["cliff", "crag", "precipice"],
    "峡谷": ["canyon", "gorge", "valley"],
    "冰川": ["glacier", "iceberg"],
    "港湾": ["harbor", "harbour", "bay", "port", "dock"],
    "海港": ["harbor", "harbour", "port", "dock", "seaport"],
    "码头": ["dock", "pier", "wharf", "harbor"],
    "海": ["sea", "ocean", "wave", "surf"],
    "湖泊": ["lake", "pond", "reservoir"],
    "河流": ["river", "stream", "creek", "brook"],
    "瀑布": ["waterfall", "falls"],
    "沙漠": ["desert", "dune", "sand"],
    "森林": ["forest", "woodland", "jungle", "rainforest", "grove"],
    "雪花": ["snow", "snowflake", "ice"],
    "冰块": ["ice", "ice cube", "iceberg"],
    "石头": ["rock", "stone", "boulder", "pebble"],
    "岩石": ["rock", "cliff", "boulder", "crag"],
    "冰山": ["iceberg"],
    "闪电": ["lightning", "thunderbolt"],
    "龙卷风": ["tornado", "whirlwind", "cyclone"],
    "沙滩": ["beach", "sand", "seashore"],
    "岛屿": ["island", "isle", "atoll"],
    "草原": ["meadow", "grassland", "prairie", "steppe", "savanna"],
    "蛋糕": ["cake", "cupcake", "chocolate cake", "sponge cake", "shortcake"],
    "面包": ["bread", "bagel", "french loaf", "toast", "pretzel", "croissant", "baguette", "loaf"],
    "汉堡": ["hamburger", "cheeseburger"],
    "披萨": ["pizza", "pizza pie"],
    "冰淇淋": ["ice cream", "icecream", "sundae", "soft-serve"],
    "糖果": ["candy", "lollipop", "sucker", "bonbon", "sweets", "gumdrop"],
    "巧克力": ["chocolate", "chocolate sauce", "cocoa"],
    "饼干": ["cookie", "biscuit", "cracker", "gingersnap"],
    "甜甜圈": ["donut", "doughnut", "cruller"],
    "薯条": ["french fries", "fries", "chips"],
    "爆米花": ["popcorn"],
    "寿司": ["sushi", "maki", "nigiri", "sashimi"],
    "面条": ["noodle", "noodles", "spaghetti", "ramen", "linguine", "vermicelli"],
    "饺子": ["dumpling", "gyoza", "potsticker", "wonton"],
    "包子": ["steamed bun", "bao", "dumpling"],
    "馒头": ["steamed bun", "mantou", "bun"],
    "月饼": ["mooncake"],
    "粽子": ["zongzi", "rice dumpling"],
    "汤圆": ["tangyuan", "glutinous rice ball", "dumpling"],
    "热狗": ["hotdog", "hot dog", "frankfurter"],
    "三明治": ["sandwich", "club sandwich", "sub", "hoagie"],
    "沙拉": ["salad", "caesar salad", "green salad"],
    "奶酪": ["cheese", "cheddar", "swiss cheese", "brie", "camembert", "parmesan"],
    "鸡蛋": ["egg", "eggs", "hen", "rooster"],
    "煎蛋": ["eggs", "omelet", "fried egg", "scrambled eggs"],
    "火腿": ["ham", "prosciutto"],
    "培根": ["bacon"],
    "薯片": ["potato chip", "crisp", "chips"],
    "棒棒糖": ["lollipop", "sucker", "candy"],
    "棉花糖": ["marshmallow", "cotton candy"],
    "华夫饼": ["waffle"],
    "可颂": ["croissant"],
    "马卡龙": ["macaron"],
    "布丁": ["pudding", "custard", "flan", "trifle"],
    "果冻": ["jelly", "jello", "gelatin"],
    "曲奇": ["cookie", "biscuit", "gingersnap"],
    "蛋挞": ["custard tart", "egg tart", "tart"],
    "年糕": ["rice cake", "mochi"],
    "饭团": ["rice ball", "onigiri"],
    "拉面": ["ramen", "noodle", "noodles"],
    "咖啡": ["coffee", "espresso", "cappuccino", "latte", "coffee mug", "coffeepot"],
    "牛奶": ["milk", "milk can", "carton"],
    "果汁": ["juice", "orange juice", "apple juice"],
    "啤酒": ["beer", "beer glass", "beer bottle", "mug", "stein"],
    "红酒": ["red wine", "wine", "wine bottle", "wineglass"],
    "可乐": ["cola", "soda", "soft drink", "pop"],
    "奶茶": ["milk tea", "bubble tea", "tea"],
    "茶": ["tea", "teapot", "teacup", "teakettle"],
    "汽水": ["soda", "soft drink", "cola"],
    "足球": ["soccer ball", "football", "ball"],
    "篮球": ["basketball", "ball"],
    "网球": ["tennis ball", "ball", "racket"],
    "羽毛球": ["badminton", "shuttlecock", "birdie", "racket"],
    "乒乓球": ["ping-pong", "table tennis", "ball"],
    "排球": ["volleyball", "ball"],
    "棒球": ["baseball", "ball", "glove", "bat"],
    "高尔夫球": ["golf ball", "golf", "ball"],
    "保龄球": ["bowling ball", "bowling"],
    "台球": ["billiards", "pool", "snooker", "ball"],
    "橄榄球": ["rugby ball", "football", "american football"],
    "冰球": ["hockey puck", "puck", "ice hockey"],
    "板球": ["cricket ball", "cricket"],
    "铅球": ["shot put", "shot"],
    "球": ["ball", "soccer ball", "basketball", "baseball", "tennis ball", "volleyball", "golf ball", "bowling ball", "cricket ball", "rugby ball", "football"],
    "书": ["book", "books", "library", "notebook", "jacket", "dust jacket", "volume", "textbook", "novel", "magazine"],
    "笔": ["pen", "ballpoint", "fountain pen", "quill", "marker", "highlighter"],
    "铅笔": ["pencil", "pencil sharpener", "pencil case"],
    "钢笔": ["fountain pen", "pen", "ballpoint"],
    "尺子": ["ruler", "yardstick", "measuring stick"],
    "橡皮": ["eraser", "rubber"],
    "剪刀": ["scissors", "shears"],
    "胶带": ["tape", "adhesive tape", "duct tape", "masking tape"],
    "订书机": ["stapler"],
    "笔记本": ["notebook", "notepad", "laptop", "journal"],
    "纸张": ["paper", "sheet", "document", "card"],
    "信封": ["envelope"],
    "画笔": ["paintbrush", "brush", "paint", "palette", "easel"],
    "蜡笔": ["crayon"],
    "颜料": ["paint", "pigment", "watercolor", "acrylic", "palette"],
    "锤子": ["hammer"],
    "扳手": ["wrench", "spanner", "monkey wrench", "pipe wrench"],
    "螺丝刀": ["screwdriver"],
    "钳子": ["pliers", "pincers", "tongs"],
    "锯子": ["saw", "handsaw", "hacksaw", "chainsaw"],
    "斧头": ["axe", "hatchet", "adze"],
    "电钻": ["drill", "power drill", "drill press"],
    "钉子": ["nail", "nail"],
    "螺丝": ["screw", "bolt", "nut"],
    "卷尺": ["tape measure", "measuring tape", "ruler"],
    "梯子": ["ladder", "step ladder"],
    "扫把": ["broom", "besom"],
    "拖把": ["mop", "mop"],
    "水桶": ["bucket", "pail"],
    "铲子": ["shovel", "spade", "trowel"],
    "锄头": ["hoe", "mattock"],
    "镰刀": ["sickle", "scythe"],
    "钥匙": ["key", "keys", "key ring", "lock"],
    "锁": ["lock", "padlock", "combination lock"],
    "灯泡": ["light bulb", "bulb", "lamp"],
    "蜡烛": ["candle", "candlestick", "candelabrum", "taper"],
    "灯笼": ["lantern", "paper lantern"],
    "风筝": ["kite"],
    "气球": ["balloon"],
    "玩具": ["toy", "doll", "teddy", "ball", "block", "puzzle"],
    "洋娃娃": ["doll", "teddy", "figurine"],
    "积木": ["block", "building block", "toy"],
    "泰迪熊": ["teddy", "teddy bear"],
    "鼓": ["drum", "drumstick", "kettledrum", "bongo"],
    "吉他": ["guitar", "acoustic guitar", "electric guitar", "banjo"],
    "钢琴": ["piano", "grand piano", "upright"],
    "小提琴": ["violin", "fiddle", "cello", "viola", "bass"],
    "喇叭": ["trumpet", "horn", "trombone", "cornet", "bugle", "tuba"],
    "笛子": ["flute", "recorder", "piccolo", "ocarina"],
    "口琴": ["harmonica", "mouth organ"],
    "铃铛": ["bell", "handbell", "cowbell", "sleigh bell"],
    "沙锤": ["maraca", "shaker"],
    "琵琶": ["lute", "pipa", "mandolin"],
    "房子": ["house", "palace", "castle", "hut", "cottage", "cabin", "lodge", "mansion", "villa", "bungalow", "igloo", "tent"],
    "城堡": ["castle", "palace", "fort", "fortress"],
    "桥": ["bridge", "suspension bridge", "viaduct"],
    "塔": ["tower", "pagoda", "minaret", "bell tower", "lighthouse"],
    "教堂": ["church", "cathedral", "chapel", "basilica", "temple"],
    "灯塔": ["lighthouse", "light tower"],
    "喷泉": ["fountain"],
    "雕像": ["statue", "sculpture", "bust", "figurine"],
    "风车": ["windmill", "wind turbine"],
    "凉亭": ["pavilion", "gazebo", "summerhouse"],
    "帐篷": ["tent", "camp"],
    "蒙古包": ["yurt", "ger"],
    "摩天轮": ["ferris wheel"],
    "旋转木马": ["carousel", "merry-go-round"],
    "滑梯": ["slide", "playground slide"],
    "秋千": ["swing", "swing set"],
    "跷跷板": ["seesaw", "teeter-totter"],
    "胡萝卜": ["carrot"],
    "土豆": ["potato", "mashed potato", "baked potato", "french fries", "fries"],
    "番茄": ["tomato", "cherry tomato"],
    "黄瓜": ["cucumber", "pickle", "gherkin"],
    "辣椒": ["pepper", "chili", "chilli", "hot pepper", "bell pepper", "capsicum"],
    "洋葱": ["onion", "red onion", "shallot", "scallion", "leek"],
    "大蒜": ["garlic"],
    "西兰花": ["broccoli", "broccoli"],
    "白菜": ["cabbage", "napa cabbage", "bok choy", "chinese cabbage"],
    "生菜": ["lettuce", "romaine", "iceberg"],
    "茄子": ["eggplant", "aubergine"],
    "南瓜": ["pumpkin", "squash", "acorn squash", "butternut", "gourd"],
    "玉米": ["corn", "maize", "ear", "popcorn", "sweet corn"],
    "花生": ["peanut", "groundnut"],
    "豆角": ["green bean", "string bean", "snap bean", "runner bean"],
    "豌豆": ["pea", "peas", "snow pea", "sugar snap"],
    "芹菜": ["celery"],
    "菠菜": ["spinach"],
    "韭菜": ["chive", "chives", "leek"],
    "姜": ["ginger"],
    "萝卜": ["radish", "daikon", "turnip", "rutabaga"],
    "红薯": ["sweet potato", "yam"],
    "山药": ["yam", "sweet potato"],
    "莲藕": ["lotus root"],
    "竹笋": ["bamboo shoot", "bamboo"],
    "苦瓜": ["bitter melon", "bitter gourd"],
    "丝瓜": ["luffa", "loofah", "sponge gourd"],
    "冬瓜": ["wax gourd", "winter melon"],
    "香菇": ["shiitake", "mushroom"],
    "金针菇": ["enoki", "mushroom"],
    "木耳": ["wood ear", "mushroom"],
    "青椒": ["bell pepper", "green pepper", "pepper"],
    "彩椒": ["bell pepper", "pepper"],
    "芦笋": ["asparagus"],
    "秋葵": ["okra"],
    "花菜": ["cauliflower", "broccoli"],
    "包菜": ["cabbage"],
    "油菜": ["bok choy", "choy sum", "rapeseed"],
    "空心菜": ["water spinach", "morning glory"],
}


def cats_of(label):
    l = label.lower()
    hits = set()
    for cat, kws in CATS.items():
        for kw in kws:
            if kw in l:
                hits.add(cat)
                break
    return hits


def classify_tiles(sprite):
    """sprite: 672x480 BGR ndarray -> returns list of 6 (tile_img, top5_labels, top5_probs)"""
    import onnxruntime as ort
    labels = load_labels(SYNSET)
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    h, w = sprite.shape[:2]
    th, tw = h // 2, w // 3
    res = []
    for i in range(6):
        col = i % 3
        row = i // 3
        t = sprite[row * th:(row + 1) * th, col * tw:(col + 1) * tw]
        img = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        x = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        x = (x - mean) / std
        x = x.transpose(2, 0, 1)[None, ...]
        out = sess.run(None, {inp.name: x})[0][0]
        top5 = np.argsort(-out)[:10]
        lab = [labels[j] for j in top5]
        prob = [float(out[j]) for j in top5]
        res.append((t, lab, prob))
    return res


def find_instruction(obj):
    if isinstance(obj, dict):
        for k in ("instruction", "Instruction", "text"):
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                return obj[k].strip()
        for v in obj.values():
            r = find_instruction(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_instruction(v)
            if r:
                return r
    return ""


def ocr_tile_img(tile_bgr):
    import subprocess, tempfile
    p = tempfile.mktemp(suffix=".png")
    cv2.imwrite(p, tile_bgr)
    try:
        out = subprocess.run(["tesseract", p, "stdout", "-l", "chi_sim+eng", "--psm", "6"],
                             capture_output=True, timeout=30)
        txt = out.stdout.decode("utf-8", "ignore")
    except Exception as e:
        log("ocr fail:", e)
        txt = ""
    try:
        os.remove(p)
    except Exception:
        pass
    return re.sub(r"\s+", "", txt)


def pick_tile_text(sprite, instruction):
    """text challenge: 包含文字[:：]X -> find tile whose OCR text contains X"""
    m = re.search(r'包含文字\s*[:：]?\s*[“"\'「]?([^”"\'」\s]+)[”"\'」]?', instruction)
    if not m:
        return 0, []
    want = m.group(1)
    if not want or want in ("图片", "的图片", "的", ":", "：", "文字"):
        return 0, []
    h, w = sprite.shape[:2]
    th, tw = h // 2, w // 3
    detail = []
    for i in range(6):
        col, row = i % 3, i // 3
        t = sprite[row * th:(row + 1) * th, col * tw:(col + 1) * tw]
        txt = ocr_tile_img(t)
        hit = want in txt
        detail.append({"tile": i + 1, "text": txt[:40], "hit": hit})
        log(f"ocr tile{i+1}: {txt[:40]} hit={hit}")
        if hit:
            return i + 1, detail
    return 0, detail


def pick_tile(res, instruction):
    """choose tile index (1-6) matching instruction; returns (idx, detail) or (0, detail)"""
    cat = INST_CAT.get(instruction, "")
    special = INST_SPECIAL.get(instruction, [])
    if not cat and not special:
        return 0, detail  # unknown instruction -> refresh instead of guessing (1/6)
    detail = []
    scores = []
    for i, (t, lab, prob) in enumerate(res):
        s = 0.0
        for j, l in enumerate(lab):
            ll = l.lower()
            wgt = 0
            for kw in special:
                if kw in ll:
                    wgt = max(wgt, 2.0)
            if wgt == 0 and cat:
                if cat == "__any__":
                    if cats_of(l):
                        wgt = 1.0
                elif cat in cats_of(l):
                    wgt = 1.0
            if wgt:
                s += wgt * (1.0 + prob[j])
        scores.append(s)
        detail.append({"tile": i + 1, "labels": lab, "score": round(s, 2)})
    mx = max(scores)
    if mx <= 0:
        return 0, detail
    # pick highest; tie -> prefer higher top-1 prob among matching
    best = scores.index(mx)
    return best + 1, detail


# ---------------- captcha interaction ----------------
def extract_instruction(frame):
    """get instruction word from frame text or prehandle responses"""
    try:
        txt = frame.inner_text("body")
    except Exception:
        txt = ""
    m = re.search(r'选择最符合描述的图片\s*[“"\'「]([^”"\'」]+)[”"\'」]', txt)
    if m:
        return m.group(1).strip()
    m2 = re.search(r'图片\s*[“"\'「]([^”"\'」]{1,8})[”"\'」]', txt)
    if m2:
        return m2.group(1).strip()
    return ""


def get_sprite_url(frame):
    try:
        style = frame.get_attribute("#slideBg", "style") or ""
    except Exception:
        style = ""
    m = re.search(r'url\(&quot;(https?://[^&]+cap_union_new_getcapbysig[^&"]*)&quot;', style)
    if not m:
        m = re.search(r'url\("?(https?://[^"]+cap_union_new_getcapbysig[^"]*)"?\)', style)
    if not m:
        # any getcapbysig url in frame html
        try:
            html = frame.content()
        except Exception:
            html = ""
        m = re.search(r'(https?://[^"\s&]+cap_union_new_getcapbysig[^"\s&]*)', html)
    if not m:
        return ""
    u = m.group(1).replace("&amp;", "&")
    return u


def click_tile(page, frame, tile_idx):
    """click the center of tile tile_idx (1-6) inside #slideBg; locator click first (iframe-safe)"""
    try:
        box = frame.locator("#slideBg").bounding_box()
    except Exception:
        box = None
    if not box:
        return False
    col = (tile_idx - 1) % 3
    row = (tile_idx - 1) // 3
    cx = (col + 0.5) * box["width"] / 3.0
    cy = (row + 0.5) * box["height"] / 2.0
    ok = False
    human_delay(0.4, 1.0)
    try:
        frame.locator("#slideBg").click(position={"x": cx, "y": cy}, timeout=8000)
        ok = True
    except Exception:
        pass
    if not ok:
        try:
            page.mouse.move(box["x"] + cx, box["y"] + cy, steps=8)
            time.sleep(random.uniform(0.15, 0.4))
            page.mouse.click(box["x"] + cx, box["y"] + cy)
            ok = True
        except Exception:
            pass
    log("clicked tile", tile_idx, "at", round(box["x"] + cx, 1), round(box["y"] + cy, 1), "via", "locator" if ok else "FAIL")
    return ok


def click_confirm(frame):
    for sel in ["#embedVerifyBtn", "button[id*='verify']", "[class*='verifyButton']"]:
        for _ in range(2):
            try:
                el = frame.query_selector(sel)
                if el:
                    human_delay(0.3, 0.8)
                    el.click(timeout=8000)
                    log("confirm clicked")
                    return True
            except Exception:
                time.sleep(1.5)
    # fallback: any button with 确定
    try:
        for b in frame.query_selector_all("button"):
            t = (b.inner_text() or "").strip()
            if t in ("确定", "确认", "完成"):
                human_delay(0.3, 0.8)
                b.click()
                log("confirm clicked (text fallback)")
                return True
    except Exception:
        pass
    return False


def refresh_challenge(frame):
    for sel in ["#embedRefreshButton", "[class*='refreshButton']", "[class*='refresh']"]:
        try:
            el = frame.query_selector(sel)
            if el:
                el.click()
                return True
        except Exception:
            pass
    return False


# ---------------- IMAP ----------------
def imap_get_code(since_ts, wait=IMAP_WAIT):
    """poll mailbox for a new Tencent verification email; return (code, url, subject)"""
    if not IMAP_PW:
        log("imap: no TC_IMAP_PW set")
        return None, None, None
    t0 = time.time()
    while time.time() - t0 < wait:
        try:
            M = imaplib.IMAP4_SSL("imap.gmail.com", 993, timeout=30)
            M.login(IMAP_USER, IMAP_PW)
            M.select("INBOX")
            typ, data = M.search(None, 'FROM', '"tencent.com"')
            ids = data[0].split()
            # newest first
            for i in reversed(ids[-6:]):
                typ, d = M.fetch(i, '(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])')
                if not d or not d[0]:
                    continue
                m = eml.message_from_bytes(d[0][1])
                date_s = m.get("Date", "")
                try:
                    dt = eml.utils.parsedate_to_datetime(date_s)
                    ts = dt.timestamp()
                except Exception:
                    continue
                if ts < since_ts - 30:
                    continue
                subj = str(make_header(decode_header(m.get("Subject", ""))))
                typ2, d2 = M.fetch(i, '(BODY.PEEK[])')
                if not d2 or not d2[0]:
                    continue
                m2 = eml.message_from_bytes(d2[0][1])
                body = ""
                if m2.is_multipart():
                    for part in m2.walk():
                        if part.get_content_type() in ("text/plain", "text/html"):
                            try:
                                body += part.get_payload(decode=True).decode("utf-8", "replace")
                            except Exception:
                                pass
                else:
                    try:
                        body = m2.get_payload(decode=True).decode("utf-8", "replace")
                    except Exception:
                        body = ""
                txt = re.sub(r"<[^>]+>", " ", body)
                txt = re.sub(r"\s+", " ", txt)
                code = None
                cm = re.search(r"验证码[：:]\s*(\d{6})", txt)
                if cm:
                    code = cm.group(1)
                urls = re.findall(r"https?://cloud\.tencent\.com[^\s\"'<>\\]*", body)
                log("imap: found mail", subj, "ts", int(ts), "code", code, "urls", len(urls))
                if code or urls:
                    M.logout()
                    return code, (urls[0] if urls else None), subj
            M.logout()
        except Exception as e:
            log("imap poll err:", e)
        time.sleep(8)
    return None, None, None


# ---------------- reset / login helpers ----------------
def fill_password_form(page, code, newpass):
    """find the set-password form (新密码 / 确认新密码) and submit. returns True on success UI."""
    # dump page for debugging
    with open(os.path.join(EVID, "reset_page.html"), "w") as f:
        f.write(page.content())
    shot(page, "reset_page.png")
    pw_inputs = page.query_selector_all("input[type='password']")
    log("password inputs:", len(pw_inputs))
    if len(pw_inputs) < 2:
        # maybe authcode input needed first (text input with 验证码 placeholder)
        txt_inputs = page.query_selector_all("input[type='text'], input:not([type])")
        for inp in txt_inputs:
            ph = inp.get_attribute("placeholder") or ""
            if "验证码" in ph or "code" in ph.lower():
                inp.fill(code)
                log("filled code input:", ph)
        pw_inputs = page.query_selector_all("input[type='password']")
    if len(pw_inputs) < 2:
        log("FAIL: no password form found")
        return False
    human_delay(0.4, 1.0)
    pw_inputs[0].fill(newpass)
    human_delay(0.3, 0.8)
    pw_inputs[1].fill(newpass)
    human_delay(0.5, 1.0)
    # submit button
    for sel in ["button[type='submit']", "button"]:
        for b in page.query_selector_all(sel):
            t = (b.inner_text() or "").strip()
            if t in ("下一步", "确定", "确认", "提交", "完成", "重置"):
                b.click()
                log("reset submit clicked:", t)
                break
        else:
            continue
        break
    # wait and check result
    time.sleep(5)
    shot(page, "reset_after.png")
    body = page.inner_text("body")
    ok = ("密码修改成功" in body) or ("修改成功" in body)
    log("reset result ok:", ok)
    with open(os.path.join(EVID, "reset_after.txt"), "w") as f:
        f.write(body[:2000])
    return ok


def do_solve(page):
    captured = {"req": [], "resp": []}
    prehandle_instructions = []

    def on_req(r):
        if "sendRecoverEmail" in r.url:
            try:
                captured["req"].append({"url": r.url, "post": r.post_data})
                log("SENDRECOVER REQ", r.url[:160], (r.post_data or "")[:300])
            except Exception:
                pass

    def on_resp(r):
        if "sendRecoverEmail" in r.url:
            try:
                body = r.text()
            except Exception:
                body = ""
            captured["resp"].append({"url": r.url, "status": r.status, "body": body[:800]})
            log("SENDRECOVER RESP", r.status, body[:400])
        if "cap_union_prehandle" in r.url or "cap_union_new_getcapbysig" in r.url:
            try:
                b = r.text()
                if len(b) < 3000 and ("instruction" in b[:500] or "{" in b[:200]):
                    fn = os.path.join(EVID, "pre_" + str(int(time.time() * 1000)) + ".json")
                    with open(fn, "w") as f:
                        f.write(b[:4000])
                try:
                    obj = json.loads(b)
                except Exception:
                    m = re.search(r"\(\s*(\{.*\})\s*\)\s*$", b, re.S)
                    obj = json.loads(m.group(1)) if m else None
                ins = find_instruction(obj) if obj else ""
                if not ins:
                    m = re.search(r'instruction["\']?\s*[:=]\s*["\']([^"\']+)', b)
                    if m:
                        ins = m.group(1)
                if ins:
                    prehandle_instructions.append(ins)
                    log("prehandle instruction:", ins)
            except Exception:
                pass

    page.on("request", on_req)
    page.on("response", on_resp)

    log("mode solve: goto recover page")
    page.goto("https://cloud.tencent.com/account/password/recover", wait_until="networkidle", timeout=90000)
    page.wait_for_selector('input[name="email"]', timeout=30000)
    shot(page, "recover_page.png")
    page.fill('input[name="email"]', EMAIL)
    human_delay(0.5, 1.2)
    page.click('button[type="submit"]')
    log("email submitted")

    # captcha popup can take up to ~90s to appear on this egress (server-side slowdown);
    # dual-channel wait: frame URL OR #tcaptcha_iframe_dy element (longer window)
    frame = None
    t0 = time.time()
    while time.time() - t0 < 150:
        frame = find_captcha_frame(page)
        if frame:
            log("captcha frame appeared after %.0fs" % (time.time() - t0))
            break
        try:
            el = page.query_selector("#tcaptcha_iframe_dy")
            if el:
                log("tcaptcha_iframe_dy element at %.0fs; waiting for frame attach" % (time.time() - t0))
                time.sleep(3)
                frame = find_captcha_frame(page)
                if frame:
                    break
        except Exception:
            pass
        if int(time.time() - t0) % 15 == 0:
            log("waiting for captcha... %.0fs" % (time.time() - t0))
        time.sleep(2)
    if not frame:
        log("FAIL: no captcha frame appeared in 150s")
        try:
            log("frames: ", [(f.url[:120]) for f in page.frames])
        except Exception:
            pass
        shot(page, "no_captcha.png")
        with open(os.path.join(EVID, "no_captcha.html"), "w") as f:
            try:
                f.write(page.content()[:50000])
            except Exception:
                pass
        return

    solved = False
    for attempt in range(1, MAX_CHALLENGES + 1):
        human_delay(1.0, 2.5)
        log(f"--- attempt {attempt} ---")
        try:
            frame = wait_captcha_frame(page, 8) or frame
        except Exception:
            pass
        instruction = extract_instruction(frame)
        if not instruction and prehandle_instructions:
            instruction = prehandle_instructions[-1]
        log("instruction:", instruction)
        sprite_url = get_sprite_url(frame)
        log("sprite_url len:", len(sprite_url))
        if not instruction or not sprite_url:
            shot(page, f"attempt_{attempt}_nocap.png")
            if attempt < MAX_CHALLENGES:
                refresh_challenge(frame)
                continue
            break
        try:
            sprite_b = dl(sprite_url)
        except Exception as e:
            log("sprite dl fail:", e)
            if attempt < MAX_CHALLENGES:
                refresh_challenge(frame)
                continue
            break
        sprite = cv2.imdecode(np.frombuffer(sprite_b, np.uint8), cv2.IMREAD_COLOR)
        if sprite is None:
            log("sprite decode fail")
            if attempt < MAX_CHALLENGES:
                refresh_challenge(frame)
                continue
            break
        cv2.imwrite(os.path.join(EVID, f"attempt_{attempt}_sprite.png"), sprite)
        # tile selection: text challenges (包含文字) use OCR; object challenges use classifier
        tile_idx = 0
        detail = []
        if "包含文字" in instruction:
            tile_idx, detail = pick_tile_text(sprite, instruction)
            log("text-tile pick:", json.dumps(detail, ensure_ascii=False)[:300])
        if tile_idx == 0:
            res = classify_tiles(sprite)
            tile_idx, detail = pick_tile(res, instruction)
        save_json(f"attempt_{attempt}_classify.json", {"instruction": instruction, "detail": detail})
        log("classify:", json.dumps(detail, ensure_ascii=False)[:600])
        if tile_idx == 0:
            log("no tile matched (unknown instruction); refresh")
            if attempt < MAX_CHALLENGES:
                refresh_challenge(frame)
                time.sleep(3)
                continue
            break
        click_tile(page, frame, tile_idx)
        time.sleep(random.uniform(3.0, 5.0))  # let the captcha JS register the selection
        shot(page, f"attempt_{attempt}_after_click.png")
        click_confirm(frame)
        # wait for outcome: verify response or sendRecoverEmail (25s window)
        t0 = time.time()
        while time.time() - t0 < 25:
            if captured["req"] or captured["resp"]:
                solved = True
                break
            # check if captcha closed/failed
            time.sleep(1)
        if solved:
            log("solve: sendRecoverEmail fired!")
            break
        # check failure markers
        try:
            body_txt = frame.inner_text("body")
        except Exception:
            body_txt = ""
        if "验证失败" in body_txt or "失败" in body_txt[:300]:
            log("attempt failed:", body_txt[:150])
        if attempt < MAX_CHALLENGES:
            if not refresh_challenge(frame):
                time.sleep(2)
            time.sleep(2)
        if solved:
            log("solve: sendRecoverEmail fired!")
            break
        # check failure markers
        try:
            body_txt = frame.inner_text("body")
        except Exception:
            body_txt = ""
        if "验证失败" in body_txt or "失败" in body_txt[:300]:
            log("attempt failed:", body_txt[:150])
        if attempt < MAX_CHALLENGES:
            if not refresh_challenge(frame):
                time.sleep(2)
    save_json("captured.json", captured)
    shot(page, "final_view.png")
    with open(os.path.join(EVID, "final_body.txt"), "w") as f:
        try:
            f.write(page.inner_text("body")[:1500])
        except Exception:
            pass

    if not (captured["req"] or captured["resp"]):
        log("FAIL: sendRecoverEmail never fired")
        return

    # success panel? wait for it
    time.sleep(2)
    shot(page, "after_send.png")
    # IMAP poll for the code
    since_ts = time.time() - 60  # email should arrive after run start; allow 60s slack
    log("polling IMAP for code...")
    code, url, subj = imap_get_code(since_ts, wait=IMAP_WAIT)
    save_json("imap_result.json", {"code": code, "url": url, "subject": subj})
    if not code:
        log("FAIL: no code received in time")
        return
    log("got code:", code)
    if not NEWPASS:
        log("FAIL: TC_NEWPASS not set")
        return

    # Now get to the reset form. The recover page after sendRecoverEmail may still hold session.
    # Try: same page state (code form may appear), or navigate with authcode param.
    filled = False
    # first: try the current page (maybe already code form)
    pw = page.query_selector_all("input[type='password']")
    if len(pw) >= 2:
        filled = fill_password_form(page, code, NEWPASS)
    if not filled:
        # try URL variants
        for u in [
            f"https://cloud.tencent.com/account/password/recover?username={EMAIL}&authcode={code}",
            f"https://cloud.tencent.com/account/password/recover?authcode={code}&username={EMAIL}",
            f"https://cloud.tencent.com/account/password/recover?authCode={code}&username={EMAIL}",
        ]:
            log("try reset url:", u)
            try:
                page.goto(u, wait_until="networkidle", timeout=60000)
            except Exception:
                pass
            time.sleep(3)
            pw = page.query_selector_all("input[type='password']")
            if len(pw) >= 2:
                filled = fill_password_form(page, code, NEWPASS)
                if filled:
                    break
    if not filled:
        log("FAIL: could not reach reset form")
        shot(page, "no_reset_form.png")
        with open(os.path.join(EVID, "no_reset_form.html"), "w") as f:
            f.write(page.content()[:30000])


def do_reset(page):
    if not RESET_URL:
        log("reset: no url")
        return
    log("goto", RESET_URL)
    page.goto(RESET_URL, wait_until="networkidle", timeout=60000)
    time.sleep(3)
    if not NEWPASS:
        log("FAIL: TC_NEWPASS not set")
        return
    code = ""
    cm = re.search(r"[?&]authcode=([^&]+)", RESET_URL)
    if cm:
        code = cm.group(1)
    fill_password_form(page, code, NEWPASS)


def do_login(page):
    log("mode login")
    page.goto("https://cloud.tencent.com/login", wait_until="networkidle", timeout=90000)
    time.sleep(3)
    shot(page, "login_page.png")
    # find email + password inputs
    email_in = None
    for sel in ["input[name='email']", "input[type='email']", "input[placeholder*='邮箱']", "input[placeholder*='账号']"]:
        try:
            el = page.query_selector(sel)
            if el:
                email_in = el
                break
        except Exception:
            pass
    pw_in = page.query_selector("input[type='password']")
    log("email input:", bool(email_in), "pw input:", bool(pw_in))
    if email_in:
        email_in.fill(EMAIL)
    if pw_in:
        human_delay(0.4, 1.0)
        pw_in.fill(NEWPASS)
    human_delay(0.5, 1.0)
    # submit
    for sel in ["button[type='submit']", "button"]:
        for b in page.query_selector_all(sel):
            t = (b.inner_text() or "").strip()
            if t in ("登录", "登 录"):
                b.click()
                log("login clicked")
                break
        else:
            continue
        break
    time.sleep(5)
    shot(page, "login_after.png")
    body = page.inner_text("body")[:1500]
    with open(os.path.join(EVID, "login_after.txt"), "w") as f:
        f.write(body)
    log("after login url:", page.url)
    # maybe email-verify step (异地登录)
    if "验证码" in body and ("邮箱" in body or "邮件" in body):
        log("email verify step detected")
        code, url, subj = imap_get_code(time.time() - 120, wait=120)
        if code:
            for inp in page.query_selector_all("input"):
                ph = inp.get_attribute("placeholder") or ""
                if "验证码" in ph or "code" in ph.lower():
                    inp.fill(code)
                    log("filled email code")
                    break
            human_delay(0.5, 1.0)
            for b in page.query_selector_all("button"):
                t = (b.inner_text() or "").strip()
                if t in ("验证", "确定", "确认", "下一步", "提交", "登录"):
                    b.click()
                    log("verify submit clicked")
                    break
            time.sleep(6)
            shot(page, "login_verified.png")
            body = page.inner_text("body")[:1500]
            with open(os.path.join(EVID, "login_verified.txt"), "w") as f:
                f.write(body)
            log("after verify url:", page.url)
    # cookies + account info
    cookies = page.context.cookies()
    save_json("cookies.json", [{"name": c["name"], "domain": c["domain"], "value": c["value"][:60]} for c in cookies])
    # try account center / console
    for u in ["https://console.cloud.tencent.com/", "https://cloud.tencent.com/account"]:
        try:
            page.goto(u, wait_until="domcontentloaded", timeout=60000)
            time.sleep(4)
            shot(page, "console_" + re.sub(r"[^a-z]", "", u) + ".png")
            txt = page.inner_text("body")[:800]
            log("page", u, "url:", page.url)
            log("body:", txt[:300])
        except Exception as e:
            log("nav fail", u, e)


def main():
    with sync_playwright() as pw:
        browser, ctx = make_ctx(pw)
        page = ctx.new_page()
        log("mode:", MODE, "email:", EMAIL, "newpass set:", bool(NEWPASS))
        if MODE == "solve":
            do_solve(page)
        elif MODE == "reset":
            do_reset(page)
        elif MODE == "login":
            do_login(page)
        else:
            log("unknown mode", MODE)
        browser.close()


if __name__ == "__main__":
    main()
