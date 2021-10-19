# ----------------- Create Shape Geometry Dataset ------------------

def polygon(output_path):
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    p = 10
    a = int(random.uniform(-p,p))
    b = int(random.uniform(-p, p))
    c = int(random.uniform(-p, p))
    d= int(random.uniform(-p, p))
    e = int(random.uniform(-p, p))
    f = int(random.uniform(-p, p))
    g = int(random.uniform(-p, p))
    h = int(random.uniform(-p, p))
    
    # --------  Square   -------------------------------------
    draw.polygon(((35+a, 35+b), (35+c, 170+d), (170+e, 170+f),
                  (170+g, 35+h)), fill="black")
    
    # --------  Triangle   -------------------------------------
    #draw.polygon(((100+a, 15+b), (15+c, 175+d), (175+e, 175+f)),
    #             fill="black")
    
    # --------  Circle   -------------------------------------
    #draw.ellipse((30 + a, 30 + b, 180 + c, 180 + d),
    #             fill="black")
    
    # --------  Hexagon   -------------------------------------
    #draw.polygon(((100 + a, 30 + b), (40 + c, 65 + d),
    #              (40 + c, 135 + e), (100 + a, 170 + f),
    #              (160 + g, 135 + e),(160 + g, 65 + d)),
    #             fill="black")
    
    image.save(output_path)
k = 1
for k in range(500):
    polygon(f"S_{k}.png")
    k+=1
