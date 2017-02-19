package data;

import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_8U;
import static org.bytedeco.javacpp.opencv_core.cvClearMemStorage;
import static org.bytedeco.javacpp.opencv_core.cvCopy;
import static org.bytedeco.javacpp.opencv_core.cvCreateImage;
import static org.bytedeco.javacpp.opencv_core.cvFlip;
import static org.bytedeco.javacpp.opencv_core.cvGetSeqElem;
import static org.bytedeco.javacpp.opencv_core.cvGetSize;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_core.cvRect;
import static org.bytedeco.javacpp.opencv_core.cvResetImageROI;
import static org.bytedeco.javacpp.opencv_core.cvSetImageROI;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;

import java.io.File;

import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacv.OpenCVFrameConverter;

public class GenderImagePreprocesing {

	public static void main(String[] args) {
		CvHaarClassifierCascade classifier = new CvHaarClassifierCascade(
				cvLoad("D:/keeplerWorkspace/Face_decetion/classifier/haarcascade_frontalface_alt.xml"));
		File folder = new File("C:\\Users\\Stefan Veselinovic\\Desktop\\genderData");
		String manDataPath = "C:\\Users\\Stefan Veselinovic\\Desktop\\processedMan\\";
		String womanDataPath = "C:\\Users\\Stefan Veselinovic\\Desktop\\processedWoman\\";
		
		OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
		OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
		int m = 0;
		int w = 0;
		if(folder.isDirectory()){
			for(File file : folder.listFiles()){
				if(!file.isDirectory()){
					Mat img = opencv_imgcodecs.imread(file.getAbsolutePath().toString());
					IplImage iplImg = toMatConverter.convertToIplImage(converter.convert(img));
					IplImage mirrorImage = iplImg.clone();
					IplImage grayImage = IplImage.create(mirrorImage.width(),
							mirrorImage.height(), IPL_DEPTH_8U, 1);

					CvMemStorage faceStorage = CvMemStorage.create();
					cvClearMemStorage(faceStorage);

					cvFlip(iplImg, mirrorImage, 1);
					cvCvtColor(mirrorImage, grayImage, CV_BGR2GRAY);
					
					CvSeq faces = opencv_objdetect.cvHaarDetectObjects(grayImage,
							classifier, faceStorage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING);
					IplImage tmp;
					IplImage imgForWritting = null;
					for (int i = 0; i < faces.total(); i++){
						CvRect rect = new CvRect(cvGetSeqElem(faces, i));
						cvSetImageROI(mirrorImage, cvRect(rect.x(), rect.y(),rect.width(),rect.height()));
						
						tmp = cvCreateImage(cvGetSize(mirrorImage),mirrorImage.depth(),mirrorImage.nChannels());
						cvCopy(mirrorImage, tmp, null);
						
				        Mat croppedMat = new Mat();
				        Mat face = toMatConverter.convert(converter.convert(tmp));
						resize(face, croppedMat, new Size(227, 227));
						imgForWritting = toMatConverter.convertToIplImage(converter.convert(croppedMat));
						
						System.out.println("--------------------------");
						System.out.println("Writing image " + file.getName() + " to ");
						
						if(file.getName().startsWith("m")){
							cvSaveImage(manDataPath + "men"+m+".png",imgForWritting);
							System.out.println(" " + manDataPath + " as " + "men"+m+".png");
							m++;
						}else if(file.getName().startsWith("wo")){
							cvSaveImage(womanDataPath + "women"+w+".png",imgForWritting);
							System.out.println(" " + womanDataPath + " as " + "women"+w+".png");
							w++;
						}
						
						System.out.println("--------------------------");
					}
					
				}
			}
			
		}
		
		
	}
	
}
