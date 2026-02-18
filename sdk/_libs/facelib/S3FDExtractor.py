from .WrapperManager import get_wrapper
import operator

class S3FDExtractor(object):
    def __init__(self, place_model_on_cpu=False, model_path=None, device_id=0):
        if place_model_on_cpu:
            device_id = -1
        self.wrapper = get_wrapper(device_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False

    def extract(self, input_image, is_bgr=True, is_remove_intersects=False):
        # wrapper.detect expects BGR
        if not is_bgr:
            # Convert RGB to BGR
            input_image = input_image[:, :, ::-1]
        
        # Detect
        # returns list of FaceStruct
        faces_structs = self.wrapper.detect(input_image)
        
        detected_faces = []
        for f in faces_structs:
            l = int(f.rect.x1)
            t = int(f.rect.y1)
            r = int(f.rect.x2)
            b = int(f.rect.y2)
            detected_faces.append([l, t, r, b])
            
        # Post-processing (Remove Intersects)
        # Mimic Python original behavior if requested
        if is_remove_intersects and len(detected_faces) > 0:
            # C++ backend already sorts by area? Yes.
            # But let's verify logic
            
            # Calculate area for sorting just in case
            # detected_faces = [ [(l,t,r,b), (r-l)*(b-t) ]  for (l,t,r,b) in detected_faces ]
            # detected_faces = sorted(detected_faces, key=operator.itemgetter(1), reverse=True )
            # detected_faces = [ x[0] for x in detected_faces]
            
            # Filter
            for i in range( len(detected_faces)-1, 0, -1):
                l1,t1,r1,b1 = detected_faces[i]
                l0,t0,r0,b0 = detected_faces[i-1]

                dx = min(r0, r1) - max(l0, l1)
                dy = min(b0, b1) - max(t0, t1)
                if (dx>=0) and (dy>=0):
                    detected_faces.pop(i)
        
        return detected_faces
