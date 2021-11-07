from PIL import Image, ImageDraw
import face_recognition as fr

image_of_barack = fr.load_image_file('image/known/barack.jpg')
barack_face_encoding = fr.face_encodings(image_of_barack)[0]

image_of_biden = fr.load_image_file('image/known/biden.jpg')
bill_face_encoding = fr.face_encodings(image_of_biden)[0]

image_of_trump = fr.load_image_file('image/known/trump.jpg')
trump_face_encoding = fr.face_encodings(image_of_trump)[0]

known_faces_encoding=[
    barack_face_encoding,
    bill_face_encoding,
    trump_face_encoding
]

known_face_names=[
    "Barack Obama",
    "Baiden",
    "Trump"
]

#load test
test_image = fr.load_image_file('./image/unknown/trumpUnk.jpg')

#find faces
face_locations = fr.face_locations(test_image)
face_encoding = fr.face_encodings(test_image,face_locations)

#convert pil format to draw
pil_image = Image.fromarray(test_image)

#Draw
draw = ImageDraw.Draw(pil_image)

for (top,right,bottom,left),face_encoding in zip(face_locations,face_encoding):
    matches = fr.compare_faces(known_faces_encoding,face_encoding)

    name="unknown Person"
    if True in matches:
        name = known_face_names[matches.index(True)]
    draw.rectangle(((left,top),(right,bottom)),outline=(0,0,0))

    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left,bottom-text_height-10),(right,bottom)),fill=(0,0,0),outline=(0,0,0))
    draw.text((left+6,bottom-text_height-5),name,fill=(255,255,255,255))

del draw

pil_image.show()