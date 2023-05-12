import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.GridPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.Stage;

public class App extends Application {
    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) throws Exception {
        GridPane pane = new GridPane();
        ImageView imageview = new ImageView();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int n = (int) (Math.random() * 3);
                if (n == 0) {
                    pane.add(new ImageView(new Image("/Users/apple/Downloads/X.jpeg")), j, i);
                    imageview.setFitWidth(10);
                    imageview.setFitHeight(10);
                }

                else if (n == 1) {
                    pane.add(new ImageView(new Image("/Users/apple/Downloads/O.png")), j, i);
                    imageview.setFitWidth(10);
                    imageview.setFitHeight(10);

                } else
                    continue;
            }
            stage.setTitle("tic-tac-toe");
            stage.setScene(new Scene(pane, 500, 500));
            stage.show();
        }

    }
}