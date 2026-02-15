**Count rejected games reasons in Rust script**
Add counting the reasons why we reject games (Elo, TimeControl, Result, Termination) to the Rust script. This will help us understand how many games we are losing at each step and adjust our filters if needed. Remember to format the file before commiting and to keep the code clean and consistent.

**Find speed improvement in Rust script**
Profile the Rust script to find bottlenecks and optimize them. This could involve using more efficient data structures, parallelizing certain parts of the code, or optimizing the way we read and write files. Make sure script is reliable, it's results are reproducible, deterministic, and easy to resume from partial results. Remember to format the file before commiting and to keep the code clean and consistent.

**Find anomalies in the data**
Think of ways to find anomalies in the data, such as games with an unusually high number of moves, games that end in a draw but have a very high Elo difference between players, or games that end with an unusual termination reason. Implement checks for these anomalies in the Rust script and log any findings. This will help us ensure the quality of our dataset and potentially adjust our filters if we find too many anomalies. Remember to format the file before commiting and to keep the code clean and consistent.

**Improve SAN -> UCI conversion**
The conversion from SAN to UCI is a critical part of our data processing pipeline. It is slow because it simulates the game move by move. Look into ways to optimize this conversion, such as caching previously seen positions and their corresponding UCI moves, or using a more efficient chess library for the conversion. Remember to format the file before commiting and t the code and run tests o keep the code clean and consistent.

**Add more tasks to the list**
Think of other tasks that could help improve our dataset or the efficiency of our data processing pipeline.

**Think of ways to improve training**
Consider ways to improve the training of our model. Good resource for this could be looking for the difference between Andrej Karpathy's MinGPT, NanoGPT and NanoChat. He uses many tricks to improve training.

**Think of a way to refactor Rust code**
Look for ways to refactor the Rust code to make it cleaner, more modular, and easier to maintain. Not sure whether we want multiple files or as it is now - one file.

**Implement unit tests for Rust code**
Add unit tests for the Rust code to ensure that each function works as expected. This will help us catch any bugs early and ensure that our code is robust. Remember to format the file before commiting and to keep the code clean and consistent.

**Add pre-commit hooks**
Set up pre-commit hooks to automatically format, lint and test the code before each commit. This will help us maintain code quality and catch any issues before they are committed to the repository. Remember to format the file before commiting and to keep the code clean and consistent.

**Setup automatic documentation generation**
Set up automatic documentation generation for both the Rust code and the Python code. It would be cool if we could generate documentation from docstrings in the code and host it somewhere like GitHub Pages. This will help us keep our documentation up to date and easily accessible. Also since we have codebase in two languages remember that we want single documentation for both of them.

**Write pipeline.sh script**
Write a `pipeline.sh` script that automates the entire data processing pipeline, from downloading games to processing them and generating the final dataset. This will make it easier to run the entire pipeline with a single command and ensure that all steps are executed in the correct order. Remember to format the file before commiting and to keep the code clean and consistent.

**Deploy the model to Lichess**
Once we have trained our model, we can deploy it to Lichess as a bot. This will allow us to test our model against real players and see how it performs in a live environment. We can also use this as an opportunity to gather more data and further improve our model.

**Setup auto-deployment to Lichess**
Set up auto-deployment of our bot to Lichess whenever we push a new version of the code. This will allow us to quickly iterate on our model and see the results in a live environment without having to manually deploy each time. Remember to format the file before commiting and to keep the code clean and consistent.

**Monitor the performance of our bot on Lichess**
Set up monitoring for our bot's performance on Lichess, such as tracking its win/loss/draw record, its performance against different types of opponents, and any issues that arise. This will help us understand how well our model is performing in a live environment and identify areas for improvement.

**Research compute opportunities**
Look into opportunities for free GPU/TPU compute for training our model. If not free make sure it's afforddable, cost effective and payed by KSI. It could be GPU credits from cloud providers, research grants, or partnerships with organizations that have access to compute resources. Also maybe University has some compute resources we could use.

**Research how model trains depending on the dataset size**
Experiment with training our model on different sizes of the dataset to see how it affects performance. This will help us understand how much data we actually need to train a good model and whether we can get away with using a smaller subset of the data to save on compute resources. We can also look into techniques like curriculum learning, where we start training on a smaller dataset and gradually increase the size as the model improves.

**Research how well model learned playing legal moves**
After training our model, we can evaluate how well it has learned to play legal moves. We can simulate games against a simple opponent and track how many times in top-k moves it suggests illegal moves. This will help us understand if our model has learned the basic rules of chess and whether we need to adjust our training data or model architecture to improve this aspect.

**Research fine-tuning on chess puzzles**
After training our model on the game data, we can fine-tune it on a dataset of chess puzzles. This will help the model learn to recognize common tactical motifs and improve its ability to suggest strong moves in tactical positions. We can evaluate the performance of the fine-tuned model on a separate set of chess puzzles to see if it has improved its tactical understanding.

**Research how well model learned chess strategy**
In addition to evaluating the model's ability to play legal moves, we can also evaluate its understanding of chess strategy. We can simulate games against a simple opponent and analyze the moves suggested by the model to see if they align with common strategic principles, such as controlling the center, developing pieces, and king safety. This will help us understand if our model has learned not just the rules of chess, but also the strategic concepts that are important for playing well.

**Research how well model learned endgames**
After training our model, we can evaluate its understanding of endgame positions. We can create a dataset of common endgame positions and evaluate the model's suggested moves in these positions to see if it has learned the key concepts of endgame play, such as king activity, pawn structure, and the importance of passed pawns. This will help us understand if our model has learned to play well in the endgame, which is a critical aspect of chess.

**Research model architecture when we give it FEN string and ask for best move**
Experiment with different model architectures that take a FEN string as input and output the best move. This will allow us to directly evaluate the model's ability to understand a chess position and suggest a strong move.

**Research interpretability of the model**
Look into techniques for interpreting the decisions made by our model. This could involve visualizing the attention weights in the model to see which parts of the input it is focusing on when making a move suggestion, or using techniques like LIME or SHAP to understand which features of the input are most influential in the model's decision-making process. This will help us understand how our model is making its decisions and potentially identify areas for improvement.

**Research improving model response for stupid moves**
We train model on games of strong players, so it might not learn how to respond to really bad moves that are not present in the training data. We can experiment with ways to improve the model's response to such moves, such as creating synthetic training data where one player is very weak and the other is strong. This will help us ensure that our model can handle a wider range of positions and respond appropriately even to suboptimal moves.
