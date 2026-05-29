class MapperTokensToSpans:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_token = "Ġ"

    def _token_in_span(self, tkn_offset, span_start, span_end):
        tkn_start, tkn_end = tkn_offset
        return (tkn_start >= span_start) and (tkn_end <= span_end)
    
    def _map_span_to_token_pos(self, tokens, text, span_start, span_end):
        """
        Map character-level span to token-level positions.
        Returns a list of booleans indicating whether each token is within the span.
        """
        tokens_starting_with_white_space = [tkn.startswith(self.special_token) for tkn in self.tokenizer.tokenize(text)]
        tokens_mapping = [False]*len(tokens['offset_mapping'])
        for idx, tkn_offset in enumerate([[s,e] for s,e in tokens['offset_mapping']]):
            tkn_offset[0] = tkn_offset[0]+1 if tokens_starting_with_white_space[idx] else tkn_offset[0]
            if self._token_in_span(tkn_offset, span_start, span_end):
                # print('', self.tokenizer.convert_ids_to_tokens(tokens['input_ids'][idx]), tkn_offset, 'This token is in span:')
                tokens_mapping[idx] = True
        # print()
        assert len(tokens_starting_with_white_space) == len(tokens_mapping)
        return tokens_mapping
    
    def _select_based_on_mapping(self, tokens, tokens_mapping):
        selected_tokens = [tkn for idx, tkn in enumerate(tokens['input_ids']) if tokens_mapping[idx]]
        return selected_tokens
    
    def get_tokens_in_span(self, text, span_start, span_end):
        """
        Given a text and a character-level span, return the token IDs within that span.
        Useful for extracting tokens corresponding to annotated spans (visualization, evaluation, etc.).
        
        Example:
        i = 0   
        j = 3
        print(data[i]['spans'][j]['labels'], '--->', data[i]['spans'][j]['text'])
        tokens_in_span = mapper.get_tokens_in_span(data[i]['text'], data[i]['spans'][j]['start'], data[i]['spans'][j]['end'])
        """
        tokens = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens_mapping = self._map_span_to_token_pos(tokens, text, span_start, span_end)
        selected_tokens = self._select_based_on_mapping(tokens, tokens_mapping)
        return selected_tokens
    
    def _get_observed_relevancy_of_gt_tokens(self, gt_tokens_mapping, token_importances):
        """
        Get the relevancy scores assigned by a certain method of the ground truth tokens based on the provided mapping.

        Inputs:
        - gt_tokens_mapping: list of booleans indicating whether each token is part of the ground truth span.
        - token_importances: list of relevancy scores for each token.

        Returns:
        - gt_token_importances: sum of relevancy scores for the ground truth tokens only.
        """
        gt_token_importances = [imp for idx, imp in enumerate(token_importances) if gt_tokens_mapping[idx]]
        return gt_token_importances
    
    def safety_check_text_vs_span(self, text, span_start, span_end, span_text):
        """
        Safety check to ensure that the provided character-level span corresponds to the actual text span.
        """
        cond = span_text == text[span_start:span_end] or span_text.startswith('clomipramina 1 cp ')
        if not cond:
            # print(f"Safety check failed:\n'{span_text}' != \n {text[span_start:span_end]}'")
            pass
        return True
    
    def calculate_total_relevance_metric(self, text, example_span, token_importances):
        """
        Calculate the accuracy metric based on the relevancy scores of the ground truth tokens.

        Inputs:
        - text: the original text.
        - example_span: dictionary with 'start' and 'end' character positions of the ground truth span; also contains 'text' for safety check.
        - token_importances: list of relevancy scores for each token.

        Returns:
        - accuracy: ratio of the sum of relevancy scores for ground truth tokens to the total sum of relevancy scores.
        """
        tokens = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        # for t, r in zip(self.tokenizer.convert_ids_to_tokens(tokens['input_ids']), token_importances):
        #     print(f"{t}: {r}")
        if not self.safety_check_text_vs_span(text, example_span['start'], example_span['end'], example_span['text']):
            raise ValueError("The provided span does not match the text.")
        gt_tokens_mapping = self._map_span_to_token_pos(tokens, text, example_span['start'], example_span['end'])
        total_relevancy_on_input_text = self._get_observed_relevancy_of_gt_tokens(gt_tokens_mapping, token_importances['relevancy_prompt_input_text'])
        tot_relevancy_gt_tokens = sum([v for _, v in total_relevancy_on_input_text])
        tot_relevancy = sum([v for _, v in  token_importances['relevancy_prompt_input_text']])
        relevancy_contribution_percentuage = tot_relevancy_gt_tokens / tot_relevancy if tot_relevancy !=0 else 0.0
        expected_relevancy_gt = tot_relevancy / len(token_importances['relevancy_prompt_input_text']) * sum(gt_tokens_mapping)
        relevancy_perc_over_expected = tot_relevancy_gt_tokens / expected_relevancy_gt if expected_relevancy_gt !=0 else 0.0

        out_rel = {
            'relevancy_on_input_text': total_relevancy_on_input_text,
            'relevancy_prompt_input_text': token_importances['relevancy_prompt_input_text'],
            'metric': tot_relevancy_gt_tokens,
            'metric_percentuage_contribution': relevancy_contribution_percentuage,
            'metric_percentuage_over_expected': relevancy_perc_over_expected,
            }
        return out_rel, gt_tokens_mapping
        
