import math
import os
import time
import json
import urllib.request
import zipfile
import requests
import Metashape
import pathlib
from predict import prediction


print(pathlib.Path().resolve())
MonitorDirectory = 'C:\\Users\\moizk\\Desktop\\Upwork\\YOLOv5-model\\imagedir\\monitor\\'      # You have to change this accordingly
RunOnce = 1    # 1= Run Once  0=Keep scanning for new tasks

# Agisoft processing settings
keypointLimit=50000 
tiepointLimit=4000 # remove smaller mesh components

def log(msg):
	print (msg)
	logfile = open(MonitorDirectory + "Process_log.txt", "a")
	logfile.write(time.strftime("%H:%M:%S") + msg + "\n")
	logfile.close()
		
def list_files(path):
    # returns a list of names (with extension, without full path) of all files 
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files 

def _get_request_payload(data={}, files={}, json_payload=False):
    """Helper method that returns the authentication token and proper content
    type depending on whether or not we use JSON payload."""
    headers = {'Authorization': 'Token {}'.format("E7F7X-Y4JG6-FL98B-4G8HB-ZK7ON")}

    if json_payload:
        headers.update({'Content-Type': 'application/json'})
        data = json.dumps(data)

    return {'data': data, 'files': files, 'headers': headers}


def processscan(scanfile):
	configfile= MonitorDirectory + scanfile
	log("JSON file: " + configfile)
	config = json.loads(open(configfile).read())
	scanid = config["scanid"]
	normaldir = config["normaldir"]
	projectdir = config["projectdir"]
	savedir = config["savedir"]
	
	try:
		SKETCHFAB_ENABLE = config["SKETCHFAB_ENABLE"]
		log("Taking JSON setting for sketchfab enable")
	except:
		log("Taking default sketchfab setting from main script")

	try:
		SKETCHFAB_DESCRIPTION = config["SKETCHFAB_DESCRIPTION"]
		log("Taking JSON setting for sketchfab description")
	except:
		log("Taking sketchfab description from main script")

	
  	# STEP 1 - Load Images
	print("\nSTEP 1 - Load Images\n")
	doc = Metashape.Document()
	doc.save(path = projectdir + "project.psx")
	# doc.clear()
	chunk = doc.addChunk()
	photos = os.listdir(normaldir) # Get the photos filenames
	photos = [os.path.join(normaldir,p) for p in photos] # Make them into a full path
	log( "Found {} photos in {}".format(len(photos), normaldir))
	if not chunk.addPhotos(photos):
		log( "ERROR: Failed to add photos: " + str(photos))
	# doc.save()


	# STEP 2 - Detect Markers
	print("\nSTEP 2 - Detect Markers\n")
	log ("Dectecting markers on non-projected images")
	chunk.detectMarkers(Metashape.TargetType.CircularTarget12bit, 50)
  	

  	# STEP 3 - Align Images
	# doc.save()
	print("\nSTEP 3 - Align Images\n")
	chunk.matchPhotos(downscale=1, keep_keypoints=True, mask_tiepoints=False, reset_matches=True, keypoint_limit=keypointLimit, tiepoint_limit=tiepointLimit)
	chunk.alignCameras(reset_alignment=True)


	# STEP 4 - Optimize Cameras and Region Reset
	# doc.save()
	print("\nSTEP 4 - Optimize Cameras and region Reset\n")
	chunk.optimizeCameras()
	chunk.resetRegion()


	# STEP 5 - Create Dense Point Cloud
	doc.save()
	print("\nSTEP 5 - Create Dense Point Cloud\n")
	chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.MildFiltering)
	chunk.buildDenseCloud(point_colors=True, point_confidence=True, keep_depth=True)


	# STEP 6 - Create 3D MESH
	# doc.save()
	print("\nSTEP 6 - Create 3D MESH\n")
	chunk.buildModel(surface_type=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation, face_count=Metashape.HighFaceCount, keep_depth=True)


	# STEP 7 - Build Texture
	print("\nSTEP 7 - Build Texture\n")
	chunk.buildUV(mapping_mode=Metashape.GenericMapping)
	chunk.buildTexture(blending=Metashape.MosaicBlending, texture_size=10240)
	'''
		Commented out, due to different model have different resolution
		better resolution have higher texture therefore longer waiting time
		it is advised to "crop" or delete any region that is not the intended object
		before manually generating texture by

		> Workflow Tab
		> "Build Texture..."

		- Texture type:	Diffuse Map
		- Source Data:	Images
		- Mapping Mode: Generic
		- Blending Mode: Mosaic
		- Texture Size Count: 4096 x 1   [ 4096 is the default value, higher reso camera suggested using 10240 x 1 or 16000 x 1 ]

		*-Advanced
		- Enable Hole Filling
		- Enable Ghosting Filter

	'''


	# STEP 8 - Create DEM
	# doc.save()
	print("\nSTEP 8 - Create DEM\n")
	chunk.buildDem(source_data=Metashape.DenseCloudData, interpolation=Metashape.EnabledInterpolation)


	# STEP 9 - Create Orthomosaic
	doc.save()
	print("\nSTEP 9 - Create Orthomosaic\n")
	chunk.buildOrthomosaic(surface_data=Metashape.ModelData, blending_mode=Metashape.MosaicBlending, fill_holes=True, ghosting_filter=False)


	# saving orthomosiac
	chunk.exportRaster(path=MonitorDirectory + 'orthophoto.png', image_format=Metashape.ImageFormatPNG, source_data = Metashape.OrthomosiacData)
	
	log("==============================================================================")
	log(" Completeted processing: " + scanid)
	log("==============================================================================")

	# running predict.py function
	prediction(MonitorDirectory + 'orthophoto.png', r'\runs\weight\best.pt')

log("Starting automatic processing engine")
Metashape.app.update()

L=0
while (L<1):
	print("Checking for new tasks..")
	Metashape.app.update()
	lst = list_files(MonitorDirectory)
	for f in lst:
		if f.endswith(".json", 0, len(f)):
			#try:
			log ("Found: " + f)
			processscan(f)
			#except:
			#	log ("General error processing scan: " + f)
			taskfile = MonitorDirectory +  f
			os.rename(taskfile, taskfile + ".done")
	#time.sleep(5)
	Metashape.app.update()
	L=L+RunOnce
	 
	 
log("The End")


