import cv2
import os
import numpy as np
import rdflib
from rdflib import URIRef, Graph
g=rdflib.Graph()
g.parse('Perception.owl')
vidcap = cv2.VideoCapture('input_video.mp4')
success,image = vidcap.read()
count = 1
currentframe=0

rdf_type=URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
w_3=URIRef('http://www.w3.org/2000/01/rdf-schema#label')
perc_cam=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#Camera_Image')
perc_ob_p=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#Object_Detection_Process')
perc_ob=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#Object_detection_Image')
perc_ob_time=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#time')
perc_occ=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#occured_in')
perc_cv_algo=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#Computer_Vision_Algorithm')
perc_cv_yolo=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#YOLO')
perc_has_op=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#has_output')
perc_has_inp=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#has_input')
SI=URIRef('http://schema.org/image')
perc_d=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#detectedobject')
perc_l=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#label')
perc_c=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#contains')
perc_hl=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#haslabel')
perc_loc_x=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#is_located_at_x')
perc_loc_y=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#is_located_at_y')
perc_y=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#objectlocation_y')
perc_x=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#objectlocation_x')


class_name=[]
objy=[]
objx=[]
def Yolo(image,class_name,objy,objx):

    def get_output_layers(net):
    
        layer_names = net.getLayerNames()
    
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers
    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

        label = str(classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open('yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                objx.append(center_x)
                objy.append(center_y)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])



    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    i_o=0
    while (i_o<len(class_ids)):
        class_name.append(classes[class_ids[i_o]])
        #objx.append((str(x[i_o])+ (str(w[i_o])/2)))
        #objy.append((str(y[i_o])+ (str(h[i_o])/2)))
        i_o=i_o+1
while success:   
  success,image = vidcap.read()
  if (count%6==0):
    currentframe=currentframe+1
    name = './input/camera_image_' + str(currentframe) + '.jpg'
    cv2.imwrite(name, image)
    perc_cam_n=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#Camera_Image_'+ str(currentframe))
    image_n=URIRef('/input/camera_image_'+ str(currentframe) + '.jpg')
    perc_time_n=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#'+ str(currentframe))
    perc_ob_p_1=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#Object_Detection_Process_'+ str(currentframe))
    g.add((perc_ob_p_1,rdf_type,perc_ob_p))
    g.add((perc_cam_n,rdf_type,perc_cam))
    g.add((perc_time_n,rdf_type,perc_ob_time))
    g.add((perc_cam_n,SI,image_n))
    g.add((perc_cam_n,perc_occ,perc_time_n))
    g.add((perc_ob_p_1,perc_has_inp,perc_cam_n))
    g.add((perc_ob_p_1,perc_has_inp,perc_cv_yolo))



    Yolo(image,class_name,objy,objx)
    out_name = './Output/Object_detection_image_' + str(currentframe) + '.jpg'
    cv2.imwrite(out_name,image)
    perc_ob_n=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#Object_detection_Image_'+ str(currentframe))
    g.add((perc_ob_n,rdf_type,perc_ob))
    g.add((perc_ob_p_1,perc_has_op,perc_ob_n))
    g.add((perc_ob_n,perc_occ,perc_time_n))
    ob_n=URIRef('./Output/Object_detection_image_' + str(currentframe) + '.jpg')
    g.add((perc_ob_n,SI,ob_n))
    g.add((perc_ob_p_1,perc_has_op,perc_ob_n))
    i=0
    while (i<len(class_name)):
      perc_d_l=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#detectedobject_'+ str(currentframe) +'_'+  str(i+1))
      perc_l_l=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#'+ str(class_name[i]))
      perc_x_l=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#'+ str(objx[i]))
      perc_y_l=URIRef('http://www.semanticweb.org/shailendran/ontologies/2021/9/untitled-ontology-7#'+ str(objy[i]))
      g.add((perc_d_l,rdf_type,perc_d))
      g.add((perc_x_l,rdf_type,perc_x))
      g.add((perc_y_l,rdf_type,perc_y))
      g.add((perc_ob_n,perc_c,perc_d_l))
      g.add((perc_l_l,rdf_type,perc_l))
      g.add((perc_d_l,perc_hl,perc_l_l))
      g.add((perc_d_l,perc_loc_x,perc_x_l))
      g.add((perc_d_l,perc_loc_y,perc_y_l))
      i=i+1
    class_name=[]
    objy=[]
    objx=[]
      
  count += 1
g.serialize(destination='Perception_out_xml.ttl', format='xml')
g.serialize(destination='Perception_out_n3.ttl', format='n3')
g.serialize(destination='Perception_out_turtle.ttl', format='turtle')
g.serialize(destination='Perception_out_xml.owl', format='xml')
g.serialize(destination='Perception_out_n3.owl', format='n3')
g.serialize(destination='Perception_out_turtle.owl', format='turtle')
g.serialize(destination='Perception_out_xml.txt', format='xml')
g.serialize(destination='Perception_out_n3.txt', format='n3')
g.serialize(destination='Perception_out_turtle.txt', format='turtle')
cv2.destroyAllWindows()
#print(count)