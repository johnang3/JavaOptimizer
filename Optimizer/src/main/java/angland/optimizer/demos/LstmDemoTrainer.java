package angland.optimizer.demos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import angland.optimizer.ngram.NGramTrainer;

public class LstmDemoTrainer {

  public static void main(String[] args) throws IOException {
    train(args[0], args[1], args[2]);
  }

  public static void train(String vocabFile, String trainFile, String contextFile)
      throws IOException {
    int vocabSize = 10000;
    int lstmSize = 200;
    int samples = 10;
    int batchSize = 10;
    int saveInterval = 1;
    double gradientClipThreshold = 0.2;
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

    System.out.println("Loading train data.");
    Files.lines(Paths.get(trainFile)).forEach(line -> {
      List<String> tokens = tokenize(line);
      List<Integer> tokenInts = new ArrayList<>();
      for (String token : tokens) {
        int tokenInt = tbm.getIdx(token);
        if (tokenInt == tbm.getUnkIdx()) {
          if (tokenInts.size() > 1) {
            trainSentences.add(tokenInts);
            tokenInts = new ArrayList<>();
          }
        } else {
          tokenInts.add(tokenInt);
        }
      }
      if (tokenInts.size() > 1) {
        trainSentences.add(tokenInts);
      }
    });
    System.out.println("Done loading train data.");
    System.out.println("Train sequences: " + trainSentences.size());

    ExecutorService es = null;
    try {
      es = Executors.newFixedThreadPool(4);
      NGramTrainer.train(es, trainSentences, new File(contextFile), vocabSize, lstmSize, batchSize,
          saveInterval, 5, samples, gradientClipThreshold);
    } finally {
      if (es != null) {
        es.shutdown();
      }
    }
  }

  public static List<String> tokenize(String line) {

    Matcher matcher = Pattern.compile("\\w+|[.,!?:\\'\\\"]").matcher(line);
    ArrayList<String> tokens = new ArrayList<>();
    while (matcher.find()) {
      tokens.add(matcher.group().toLowerCase());
    }
    return tokens;
  }
}
