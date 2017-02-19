package main;

import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_8U;
import static org.bytedeco.javacpp.opencv_core.cvClearMemStorage;
import static org.bytedeco.javacpp.opencv_core.cvCreateImage;
import static org.bytedeco.javacpp.opencv_core.cvFlip;
import static org.bytedeco.javacpp.opencv_core.cvGetSeqElem;
import static org.bytedeco.javacpp.opencv_core.cvGetSize;
import static org.bytedeco.javacpp.opencv_core.cvPoint;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;
import static org.bytedeco.javacpp.opencv_imgproc.cvDrawRect;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.CvScalar;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_ml.ANN_MLP;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.UByteBufferIndexer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_dnn.*;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class Controller {

	@FXML
	private Button camButtonStart;
	@FXML
	private Button camButtonGender;
	@FXML
	private Button camButtonAge;
	
	@FXML
	private ImageView imageFrame;

	private CvHaarClassifierCascade classifier;
	private boolean cameraActive = false;
	private boolean predictAgeActive = false;
	private boolean predictGenderActive = false;
	
	private OpenCVFrameGrabber grabber;
	private OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
	private OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
	private Java2DFrameConverter paintConverter = new Java2DFrameConverter();
	private ScheduledExecutorService timer;
	
	
	private Net genderNet;
	private Net ageNet;
	
	private static final String[] AGES = new String[]{"0-2", "4-6", "8-13", "13-18", "19-24", "25-33", "35-42", "45-"};
													// a       b      c        d        e         f       g       h
    private enum Gender {
        MALE,
        FEMALE,
        NOT_RECOGNIZED
    }
    
	public void init() {
		classifier = new CvHaarClassifierCascade(
				cvLoad("classifier/haarcascade_frontalface_alt.xml"));
		if (classifier.isNull()) {
			System.out.println("Error loading classifier");
			System.exit(1);
		}
		genderNet = new Net();
		ageNet = new Net();
		
		File genProtobuf = new File("prototxts/deploy_gender.prototxt");
        File genCaffeModel = new File("caffeModels/gender.caffemodel");
        Importer genderImporter = createCaffeImporter(genProtobuf.getAbsolutePath(), genCaffeModel.getAbsolutePath());
        genderImporter.populateNet(genderNet);
        genderImporter.close();
        
        File ageProtobuf = new File("prototxts/deploy_age.prototxt");
        File ageCaffeModel = new File("caffeModels/age.caffemodel");
        Importer ageImporter = createCaffeImporter(ageProtobuf.getAbsolutePath(), ageCaffeModel.getAbsolutePath());
        ageImporter.populateNet(ageNet);
        ageImporter.close();
        this.camButtonAge.setDisable(true);
        this.camButtonGender.setDisable(true);
	}
	
	public void startGenderPrediction(){
		if(this.cameraActive){
			this.predictGenderActive = !this.predictGenderActive;
			if(this.camButtonGender.getText().equals("Start predicting gender")){
				this.camButtonGender.setText("Stop predicting gender");
			}else{
				this.camButtonGender.setText("Start predicting gender");
			}
		}
	}
	
	public void startAgePrediction(){
		if(this.cameraActive){
			this.predictAgeActive = !this.predictAgeActive;
			if(this.camButtonAge.getText().equals("Start predicting age")){
				this.camButtonAge.setText("Stop predicting age");
			}else{
				this.camButtonAge.setText("Start predicting age");
			}
		}
	}

	public void startCamera() throws Exception {
		Loader.load(opencv_objdetect.class);
		imageFrame.setFitWidth(900);
		imageFrame.setPreserveRatio(true);

		if (!cameraActive){
			this.cameraActive = true;
			this.camButtonAge.setDisable(false);
	        this.camButtonGender.setDisable(false);
			
		
			grabber = new OpenCVFrameGrabber(0);
			grabber.start();
			
			Runnable runnable = new Runnable() {	
				@Override
				public void run() {
	
					try {
						imageFrame.setImage(getImage());
						if(!cameraActive) imageFrame.setImage(null);
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			};
			this.timer = Executors.newSingleThreadScheduledExecutor();
			this.timer.scheduleAtFixedRate(runnable, 0, 33, TimeUnit.MILLISECONDS);
			this.camButtonStart.setText("Stop Camera");
		}else {
			this.camButtonAge.setDisable(true);
	        this.camButtonGender.setDisable(true);
			this.cameraActive = false;
			this.camButtonStart.setText("Start Camera");
			
			try
			{
				this.timer.shutdown();
				this.timer.awaitTermination(30, TimeUnit.MILLISECONDS);
				this.imageFrame.setImage(null);
			}
			catch (InterruptedException e)
			{
				e.printStackTrace();
			}
			grabber.stop();
			this.imageFrame.setImage(null);
			
		}
	}
	
	private Image getImage() throws Exception{
		Frame frame = grabber.grab();
		IplImage grabbedImg = converter.convert(frame);

		IplImage mirrorImage = grabbedImg.clone();
		IplImage grayImage = IplImage.create(mirrorImage.width(),
				mirrorImage.height(), IPL_DEPTH_8U, 1);

		CvMemStorage faceStorage = CvMemStorage.create();
		BufferedImage bufferedImage;

		cvClearMemStorage(faceStorage);
		cvFlip(grabbedImg, mirrorImage, 1);
		cvCvtColor(mirrorImage, grayImage, CV_BGR2GRAY);
		
		findAndDrawFaces(classifier, faceStorage, CvScalar.GREEN,
				grayImage, mirrorImage, frame);
		bufferedImage = paintConverter.convert(converter
				.convert(mirrorImage));
		
		return SwingFXUtils.toFXImage(bufferedImage,null);
	}

	private void findAndDrawFaces(CvHaarClassifierCascade classifier,
			CvMemStorage faceStorage, CvScalar color, IplImage grayImage,
			IplImage mirrorImage, Frame frame) {

		CvSeq faces = opencv_objdetect.cvHaarDetectObjects(grayImage,
				classifier, faceStorage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING);
		IplImage tmp;
		for (int i = 0; i < faces.total(); i++) {
			CvRect rect = new CvRect(cvGetSeqElem(faces, i));
			opencv_imgproc.cvDrawRect(mirrorImage, cvPoint(rect.x(), rect.y()),
					cvPoint(rect.x() + rect.width(), rect.y() + rect.height()),
					color, 1, 1, 0);
			
			cvSetImageROI(mirrorImage, cvRect(rect.x(), rect.y(),rect.width(),rect.height()));
			tmp = cvCreateImage(cvGetSize(mirrorImage),mirrorImage.depth(),mirrorImage.nChannels());
			cvCopy(mirrorImage, tmp, null);
			cvResetImageROI(mirrorImage);

	        Mat croppedMat = new Mat();
	        Mat face = toMatConverter.convert(converter.convert(tmp));
			resize(face, croppedMat, new Size(256, 256));
			
			normalize(croppedMat, croppedMat, 0, Math.pow(2, frame.imageDepth), NORM_MINMAX, -1, null);
			
			Gender gender = null;				
			if(this.predictGenderActive)
				gender = predictGender(toMatConverter.convert(converter.convert(tmp)),frame);
			String age = "";
			if(this.predictAgeActive)
				age = predictAge(toMatConverter.convert(converter.convert(tmp)),frame);
			
			Mat mat = null;
			if(this.predictAgeActive || this.predictGenderActive)
				mat = toMatConverter.convert(converter.convert(mirrorImage));
			
			if(this.predictAgeActive && this.predictGenderActive){
				if(gender.equals(Gender.MALE)){
					opencv_imgproc.putText(mat, "Male "+age, new Point(rect.x(), rect.y()), opencv_core.FONT_ITALIC  , 0.55 ,new  Scalar(255.0, 0.0, 0.0, 1.0));
				}else{
					opencv_imgproc.putText(mat, "Female "+age, new Point(rect.x(), rect.y()), opencv_core.FONT_ITALIC  , 0.55 ,new  Scalar(255.0, 0.0, 0.0, 1.0));
				}
			}else{
				if(this.predictAgeActive){
					opencv_imgproc.putText(mat, age, new Point(rect.x(), rect.y()), opencv_core.FONT_ITALIC  , 0.55 ,new  Scalar(255.0, 0.0, 0.0, 1.0));
					
				}
				if(this.predictGenderActive){
					if(gender.equals(Gender.MALE)){
						opencv_imgproc.putText(mat, "Male ", new Point(rect.x(), rect.y()), opencv_core.FONT_ITALIC  , 0.55 ,new  Scalar(255.0, 0.0, 0.0, 1.0));
					}else{
						opencv_imgproc.putText(mat, "Female ", new Point(rect.x(), rect.y()), opencv_core.FONT_ITALIC  , 0.55 ,new  Scalar(255.0, 0.0, 0.0, 1.0));
					}
				}
			}
		}

	}
	
	public Gender predictGender(Mat face, Frame frame) {
        Mat croppedMat = new Mat();
		resize(face, croppedMat, new Size(256, 256));
		
		normalize(croppedMat, croppedMat, 0, Math.pow(2, frame.imageDepth), NORM_MINMAX, -1, null);
		
		Blob inputBlob = new Blob(croppedMat);
		genderNet.setBlob(".data", inputBlob);
		genderNet.forward();
		Blob prob = genderNet.getBlob("prob");

		Indexer indexer = prob.matRefConst().createIndexer();
		if (indexer.getDouble(0, 0) > indexer.getDouble(0, 1)) {
			return Gender.MALE;
		} else {
		    return Gender.FEMALE;
		}
	}

    public String predictAge(Mat face, Frame frame) {
        Mat resizedMat = new Mat();
		resize(face, resizedMat, new Size(256, 256));
		normalize(resizedMat, resizedMat, 0, Math.pow(2, frame.imageDepth), NORM_MINMAX, -1, null);

		Blob inputBlob = new Blob(resizedMat);
		ageNet.setBlob(".data", inputBlob);
		ageNet.forward();
		Blob prob = ageNet.getBlob("prob");

		DoublePointer pointer = new DoublePointer(new double[1]);
		Point max = new Point();
		minMaxLoc(prob.matRefConst(), null, pointer, null, max, null);
		return AGES[max.x()];
		
    }

}
