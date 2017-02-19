package main;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;


public class Frame extends Application{

	public static void main(String[] args) {
		launch(args);
	}
	@Override
	public void start(Stage stage) throws Exception {

		FXMLLoader loader = new FXMLLoader(getClass().getResource("FaceDetection.fxml"));
		BorderPane pane = (BorderPane)loader.load();
		
		Scene scene = new Scene(pane, 1000, 800);
		
		stage.setTitle("SOFT 2017 - Face Detection");
		stage.setScene(scene);
		stage.show();
		
		Controller controller = loader.getController();
		controller.init();
		
	}

}
