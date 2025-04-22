

class Preprocessor():
    def __init__(self, tokenizer, num_proc=4, batch_size=16):
        self.tokenizer = tokenizer
        self.num_proc = num_proc
        self.batch_size = batch_size

    def _preprocess_interactive(self, examples):
        queries = [self.documents[doc_id]['title'] for doc_id in examples['query']]
        candidates = [self.documents[doc_id]['title'] for doc_id in examples['candidate']]

        result = self.tokenizer(
            queries,
            candidates,
            padding=True,
            truncation=True,
            return_attention_mask=True
        )

        result['labels'] = examples['relevance']

        return result

    def preprocess(self, queries, documents):
        self.documents = documents

        dataset = queries.map(
            self._preprocess_interactive,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            remove_columns=queries.column_names
        )

        dataset.set_format(
            type='torch', 
            columns=[
                'input_ids', 
                'attention_mask', 
                'token_type_ids',
                'labels'
            ]
        )
        
        return dataset
