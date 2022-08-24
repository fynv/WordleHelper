#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>

void judge(const std::string& truth, const std::string& guess, int feedback[5])
{
	unsigned char used[5] = { 0,0,0,0,0 };

	for (int i = 0; i < 5; i++)
	{
		if (guess[i] == truth[i])
		{
			feedback[i] = 2;
			used[i] = 1;
		}
		else
		{
			feedback[i] = 0;
		}
	}

	for (int i = 0; i < 5; i++)
	{
		if (feedback[i] == 0)
		{
			for (int j = 0; j < 5; j++)
			{
				if (used[j] == 0 && guess[i] == truth[j])
				{
					feedback[i] = 1;
					used[j] = 1;
					break;
				}
			}
		}
	}
}

int encode(const int feedback[5])
{
	return (((feedback[0] * 10 + feedback[1]) * 10 + feedback[2]) * 10 + feedback[3]) * 10 + feedback[4];
}


int main()
{
	std::vector<std::string> words;
	std::vector<std::string> alloweds;
	{
		FILE* fp = fopen("wordle.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			words.push_back(word);
		}
		fclose(fp);
	}

	{
		FILE* fp = fopen("wordle-allowed-guesses.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			alloweds.push_back(word);
		}
		fclose(fp);
	}

	while (true)
	{
		std::string best;
		if (words.size() > 1)
		{
			std::vector<double> entropies(alloweds.size());
			#pragma omp parallel for			
			for (int i = 0; i < (int)alloweds.size(); i++)
			{
				std::string guess = alloweds[i];
				std::unordered_map<int, int> counts;

				for (size_t j = 0; j < words.size(); j++)
				{
					std::string truth = words[j];
					int feedback[5];
					judge(truth, guess, feedback);
					int code = encode(feedback);
					int& count = counts[code];
					count++;
				}

				double count_all = (double)words.size();
				double entropy = 0.0;
				auto iter = counts.begin();
				while (iter != counts.end())
				{
					double count = (double)iter->second;
					double p = count / count_all;
					double info = -log(p) / log(2.0);
					entropy += p * info;
					iter++;
				}

				entropies[i] = entropy;
			}

			double max_entropy = 0.0;
			for (size_t i = 0; i < alloweds.size(); i++)
			{
				double entropy = entropies[i];
				if (entropy > max_entropy)
				{
					max_entropy = entropy;
					best = alloweds[i];
				}
			}
		}
		else
		{
			best = words[0];
		}

		printf("%s\n", best.c_str());

		int feedback[5];
		while (true)
		{
			char line[100];
			scanf("%s", line);
			if (strlen(line) >= 5)
			{
				for (int i = 0; i < 5; i++)
				{
					if (line[i] == '1') feedback[i] = 1;
					else if (line[i] == '2') feedback[i] = 2;
					else feedback[i] = 0;
				}
				break;
			}
		}

		bool match = true;
		for (int i = 0; i < 5; i++)
		{
			if (feedback[i] != 2)
			{
				match = false;
				break;
			}
		}
		if (match) break;

		for (size_t i = 0; i < words.size(); i++)
		{
			std::string word = words[i];

			bool remove = false;

			int feedback2[5];
			judge(word, best, feedback2);
			for (int j = 0; j < 5; j++)
			{
				if (feedback2[j] != feedback[j])
				{
					remove = true;
					break;
				}
			}

			if (remove)
			{
				words.erase(words.begin() + i);
				i--;
			}
		}
	}
	
	return 0;
}

/*int main()
{
	std::vector<std::string> all_words;
	std::vector<std::string> alloweds;
	{
		FILE* fp = fopen("wordle.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			all_words.push_back(word);
		}
		fclose(fp);
	}

	{
		FILE* fp = fopen("wordle-allowed-guesses.txt", "r");
		char line[100];
		while (fgets(line, 100, fp))
		{
			char word[100];
			sscanf(line, "%s", word);
			alloweds.push_back(word);
		}
		fclose(fp);
	}

	double ave_guesses = 0.0;
	for (size_t i_truth = 0; i_truth < all_words.size(); i_truth++)
	{
		std::string truth_i = all_words[i_truth];

		std::vector<std::string> words = all_words;

		int guess_count = 0;
		while (true)
		{
			std::string best;
			if (words.size() > 1)
			{	
				std::vector<double> entropies(alloweds.size());
				#pragma omp parallel for
				for (int i = 0; i < (int)alloweds.size(); i++)
				{
					std::string guess = alloweds[i];
					std::unordered_map<int, int> counts;

					for (size_t j = 0; j < words.size(); j++)
					{
						std::string truth = words[j];
						int feedback[5];
						judge(truth, guess, feedback);
						int code = encode(feedback);
						int& count = counts[code];
						count++;
					}

					double count_all = (double)words.size();
					double entropy = 0.0;
					auto iter = counts.begin();
					while (iter != counts.end())
					{
						double count = (double)iter->second;
						double p = count / count_all;
						double info = -log(p) / log(2.0);
						entropy += p * info;
						iter++;
					}

					entropies[i] = entropy;
				}

				double max_entropy = 0.0;
				for (size_t i = 0; i < alloweds.size(); i++)
				{
					double entropy = entropies[i];
					if (entropy > max_entropy)
					{
						max_entropy = entropy;
						best = alloweds[i];
					}
				}
			}
			else
			{
				best = words[0];
			}

			int feedback[5];
			judge(truth_i, best, feedback);
			
			guess_count++;

			bool match = true;
			for (int i = 0; i < 5; i++)
			{
				if (feedback[i] != 2)
				{
					match = false;
					break;
				}
			}
			if (match) break;

			for (size_t i = 0; i < words.size(); i++)
			{
				std::string word = words[i];

				bool remove = false;

				int feedback2[5];
				judge(word, best, feedback2);
				for (int j = 0; j < 5; j++)
				{
					if (feedback2[j] != feedback[j])
					{
						remove = true;
						break;
					}
				}

				if (remove)
				{
					words.erase(words.begin() + i);
					i--;
				}
			}
		}

		ave_guesses += (double)guess_count;

		printf("%s %d\n", truth_i.c_str(), guess_count);
	}
	ave_guesses /= (double)all_words.size();

	printf("%f\n", ave_guesses);

	return 0;
}*/
