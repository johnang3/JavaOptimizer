package angland.optimizer.demos;

import static angland.optimizer.demos.DemoConstants.vocabSize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import angland.optimizer.ngram.NGramPredictor;
import angland.optimizer.nn.RnnCellTemplate;
import angland.optimizer.saver.StringContext;
import angland.optimizer.var.Context;
import angland.optimizer.var.ContextTemplate;
import angland.optimizer.var.scalar.IScalarValue;
import angland.optimizer.var.scalar.StreamingSum;

public class LstmDemoValidator {

  public static void main(String[] args) throws FileNotFoundException, IOException {
    printLoss(args[0], args[1], args[2]);
  }

  public static void printLoss(String vocabFile, String validateDir, String contextFile)
      throws FileNotFoundException, IOException {
    int samples = 80;
    List<String> vocabTokens = new ArrayList<>();
    vocabTokens.add("<unk>");
    try (FileReader fr = new FileReader(vocabFile); BufferedReader br = new BufferedReader(fr)) {
      String line = null;
      while ((line = br.readLine()) != null && vocabTokens.size() < vocabSize) {
        vocabTokens.add(line.split(" ")[0]);
      }
    }
    System.out.println("Vocabulary size " + vocabTokens.size());
    TokenBiMap tbm = new TokenBiMap(vocabTokens, "<unk>");


    List<List<Integer>> trainSentences = new ArrayList<>();
    List<File> filesInFolder =
        Files.walk(Paths.get(validateDir)).filter(Files::isRegularFile).map(Path::toFile)
            .collect(Collectors.toList());

    System.out.println("Loading validation data.");
    for (File file : filesInFolder) {
      Files.lines(Paths.get(file.getAbsolutePath())).forEach(line -> {
        List<String> tokens = TokenBiMap.tokenize(line);
        List<Integer> tokenInts = new ArrayList<>();
        for (String token : tokens) {
          int tokenInt = tbm.getIdx(token);
          if (tokenInt == tbm.getUnkIdx()) {
            if (tokenInts.size() > 5) {
              trainSentences.add(tokenInts);
            }
            tokenInts = new ArrayList<>();
          } else {
            tokenInts.add(tokenInt);
          }
        }
        if (tokenInts.size() > 5) {
          trainSentences.add(tokenInts);
        }
      });
    }
    System.out.println("Done loading train data.");
    System.out.println("Validation sequences: " + trainSentences.size());
    RnnCellTemplate template = DemoConstants.getTemplate(true);
    ContextTemplate<String> contextTemplate =
        new ContextTemplate<>(NGramPredictor.getKeys(vocabSize, template).collect(
            Collectors.toList()));
    Context<String> context = contextTemplate.createContext(StringContext.loadContext(contextFile));

    NGramPredictor predictor = new NGramPredictor(vocabSize, template, context, true);
    List<IScalarValue<String>> losses =
        trainSentences.stream().map(t -> predictor.getLoss(t, samples))
            .collect(Collectors.toList());
    IScalarValue<String> loss =
        new StreamingSum<>(losses).divide(IScalarValue.constant(losses.size()));
    System.out.println("Loss: " + loss.value());
  }
}
