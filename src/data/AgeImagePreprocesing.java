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
import static org.bytedeco.javacpp.opencv_core.cvSetImageROI;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvSaveImage;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;

import java.io.File;

import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacv.OpenCVFrameConverter;

public class AgeImagePreprocesing {

	public static void main(String[] args) {
		CvHaarClassifierCascade classifier = new CvHaarClassifierCascade(
				cvLoad("D:/keeplerWorkspace/Face_decetion/classifier/haarcascade_frontalface_alt.xml"));
		File folder = new File("C:\\Users\\Stefan Veselinovic\\Desktop\\ageData");
		String aDataPath = "C:\\Users\\Stefan Veselinovic\\Desktop\\a\\";
		String bDataPath = "C:\\Users\\Stefan Veselinovic\\Desktop\\b\\";
		String cDataPath = "C:\\Users\\Stefan Veselinovic\\Desktop\\c\\";
		String dDataPath = "C:\\Users\\Stefan Veselinovic\\Desktop\\d\\";
		String eDataPath = "C:\\Users\\Stefan Veselinovic\\Desktop\\e\\";
		String fDataPath = "C:\\Users\\Stefan Veselinovic\\Desktop\\f\\";
		String gDataPath = "C:\\Users\\Stefan Veselinovic\\Desktop\\g\\";
		
		OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
		OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
		int a = 0;
		int b = 0;
		int c = 0;
		int d = 0;
		int e = 0;
		int f = 0;
		int g = 0;
		
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
						
						switch(file.getName().charAt(0)){
							case 'a': {
								cvSaveImage(aDataPath + "a"+a+".png",imgForWritting);
								System.out.println(" " + aDataPath + " as " + "a"+a+".png");
								a++;
								break; 
							}
							case 'b': {
								cvSaveImage(bDataPath + "b"+b+".png",imgForWritting);
								System.out.println(" " + bDataPath + " as " + "b"+b+".png");
								b++;
								break; 
							}
							case 'c': {
								cvSaveImage(cDataPath + "c"+c+".png",imgForWritting);
								System.out.println(" " + cDataPath + " as " + "c"+c+".png");
								c++;
								break; 
							}
							case 'd': {
								cvSaveImage(dDataPath + "d"+d+".png",imgForWritting);
								System.out.println(" " + dDataPath + " as " + "d"+d+".png");
								d++;
								break; 
							}
							case 'e': {
								cvSaveImage(eDataPath + "e"+e+".png",imgForWritting);
								System.out.println(" " + eDataPath + " as " + "e"+e+".png");
								e++;
								break; 
							}
							case 'f': {
								cvSaveImage(fDataPath + "f"+f+".png",imgForWritting);
								System.out.println(" " + fDataPath + " as " + "f"+f+".png");
								f++;
								break; 
							}
							case 'g': {
								cvSaveImage(gDataPath + "g"+g+".png",imgForWritting);
								System.out.println(" " + gDataPath + " as " + "g"+g+".png");
								g++;
								break; 
							}
						}
						
						System.out.println("--------------------------");
					}
					
				}
			}
			
		}
		

	}
}
