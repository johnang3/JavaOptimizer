package angland.optimizer.demos;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.Collectors;

import angland.optimizer.ngram.NGramPredictor;
import angland.optimizer.saver.StringContext;
import angland.optimizer.var.IndexedKey;

public class LstmDemoInteractive {

  public static void main(String[] args) throws FileNotFoundException, IOException {
    int vocabSize = 10000;
    int lstmSize = 40;
    double gradientClipThreshold = 0.2;
    List<String> vocabTokens = new ArrayList<>();
    vocabTokens.add("<unk>");
    try (FileReader fr = new FileReader(args[0]); BufferedReader br = new BufferedReader(fr)) {
      String line = null;
      while ((line = br.readLine()) != null && vocabTokens.size() < vocabSize) {
        vocabTokens.add(line.split(" ")[0]);
      }
    }
    System.out.println("Vocabulary size " + vocabTokens.size());
    TokenBiMap tbm = new TokenBiMap(vocabTokens, "<unk>");
    System.out.println("Loading model. ");
    Map<IndexedKey<String>, Double> context = StringContext.loadContext(args[1]);
    NGramPredictor predictor =
        new NGramPredictor(vocabSize, lstmSize, context, gradientClipThreshold, true);
    try (Scanner scan = new Scanner(System.in);) {
      System.out.println("Type and press enter to view predicted next tokens.");
      while (true) {
        String line = scan.nextLine();
        List<String> tokens = TokenBiMap.tokenize(line);
        List<Integer> tokenInts = tokens.stream().map(tbm::getIdx).collect(Collectors.toList());
        List<Integer> response = predictor.predictNext(tokenInts, 10);
        List<String> responseTokens =
            response.stream().map(tbm::getToken).collect(Collectors.toList());
        responseTokens.forEach(t -> System.out.print(t + " "));
        System.out.println();
      }
    }

  }
}
