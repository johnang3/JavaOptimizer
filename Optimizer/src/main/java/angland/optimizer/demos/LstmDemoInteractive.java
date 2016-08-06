package angland.optimizer.demos;

import static angland.optimizer.demos.DemoConstants.vocabSize;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

import angland.optimizer.ngram.NGramPredictor;
import angland.optimizer.nn.RnnCellTemplate;
import angland.optimizer.saver.StringContext;
import angland.optimizer.var.Context;
import angland.optimizer.var.ContextTemplate;

public class LstmDemoInteractive {

  public static void main(String[] args) throws FileNotFoundException, IOException {
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
    RnnCellTemplate template = DemoConstants.getTemplate(true);
    ContextTemplate<String> contextTemplate =
        new ContextTemplate<>(NGramPredictor.getKeys(vocabSize, template).collect(
            Collectors.toList()));
    Context<String> context = contextTemplate.createContext(StringContext.loadContext(args[1]));

    NGramPredictor predictor = new NGramPredictor(vocabSize, template, context, true);
    try (Scanner scan = new Scanner(System.in);) {
      System.out.println("Type and press enter to view predicted next tokens.");
      while (true) {
        String line = scan.nextLine();
        List<String> tokens = TokenBiMap.tokenize(line);
        List<Integer> tokenInts = tokens.stream().map(tbm::getIdx).collect(Collectors.toList());
        List<Integer> response = predictor.predictNext(tokenInts, 10, tbm.getIdx("<unk>"));
        List<String> responseTokens =
            response.stream().map(tbm::getToken).collect(Collectors.toList());
        responseTokens.forEach(t -> System.out.print(t + " "));
        System.out.println();
      }
    }

  }
}
