import java.util.Random;

public class temp {
    public static void main(String[] args) {
        Random random = new Random();
        String plate = "";
        for (int i = 0; i < 3; i++) {
            plate += (char) (random.nextInt(26) + 'A');
        }
        for (int i = 0; i < 4; i++) {
            plate += random.nextInt(10);
        }
        System.out.println(plate);
    }
}