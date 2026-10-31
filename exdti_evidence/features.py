"""Observable model inputs. Oracle claim assessments are deliberately absent."""
import json
import torch


def clean_sequence(sequence):
    return "".join("".join(line.split()) for line in sequence.splitlines()
                   if not line.lstrip().startswith(">"))


def pair_text(record):
    # Names, entity IDs, action labels and literature are not predictor inputs.
    return "DRUG " + record["drug"]["smiles"] + " TARGET " + clean_sequence(record["target"]["sequence"])


def assertion_text(assertion):
    # Put semantic content before potentially long entity identifiers.
    keys = ("action", "polarity", "condition", "drug_id", "target_id")
    return json.dumps({key: assertion.get(key) for key in keys}, ensure_ascii=False)


def claim_text(claim):
    # A claim is a positive action query, never a supplied state or polarity.
    keys = ("action", "condition", "drug_id", "target_id")
    return "CLAIM " + json.dumps({key: claim.get(key) for key in keys}, ensure_ascii=False) + " ID " + claim["claim_id"]


class FeatureBuilder:
    def __init__(self, tokenizer, config):
        self.tokenizer, self.config = tokenizer, config

    def tokens(self, texts, max_length, device):
        result = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
        return {k: v.to(device) for k, v in result.items() if k in ("input_ids", "attention_mask")}

    def pair_tokens(self, record, device):
        """Budget both modalities independently so long SMILES cannot erase protein.

        Budgets count tokenizer tokens, not residues. A single final EOS follows
        DRUG-prefix + drug-content + TARGET-prefix + protein-content. The protein
        setting is named ``protein_max_length`` because ``target_max_length`` is
        the existing decoder supervision budget.
        """
        drug_prefix = self.tokenizer.encode("DRUG ", add_special_tokens=False)
        protein_prefix = self.tokenizer.encode(" TARGET ", add_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Pair tokenization requires an EOS token")
        available = self.config.get("pair_max_length", 384) - len(drug_prefix) - len(protein_prefix) - 1
        if available < 2:
            raise ValueError("pair_max_length must reserve at least one token for each modality")
        drug_budget = self.config.get("drug_max_length")
        protein_budget = self.config.get("protein_max_length")
        if drug_budget is None and protein_budget is None:
            drug_budget = max(1, available // 3)
            protein_budget = available - drug_budget
        elif drug_budget is None:
            drug_budget = available - protein_budget
        elif protein_budget is None:
            protein_budget = available - drug_budget
        if min(drug_budget, protein_budget) < 1 or drug_budget + protein_budget > available:
            raise ValueError("Drug/protein token budgets must be positive and fit pair_max_length including prefixes/EOS")
        smiles = record["drug"]["smiles"]
        sequence = clean_sequence(record["target"]["sequence"])
        if not smiles.strip() or not sequence:
            raise ValueError("Both SMILES and a headerless protein sequence are required")
        drug = self.tokenizer.encode(smiles, add_special_tokens=False, truncation=True, max_length=drug_budget)
        protein = self.tokenizer.encode(sequence, add_special_tokens=False, truncation=True, max_length=protein_budget)
        if not drug or not protein:
            raise ValueError("Both drug and protein must produce at least one content token")
        ids = torch.tensor([drug_prefix + drug + protein_prefix + protein + [eos]], dtype=torch.long, device=device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def __call__(self, record, evidence, device):
        if evidence.get("record_id") != record["record_id"]:
            raise ValueError("Evidence belongs to a different record")
        sources = evidence["sources"]
        # Fail on truncating sources: silently dropping a source changes the support oracle.
        if len(sources) > self.config.get("max_sources", 32):
            raise ValueError("Too many sources; raise max_sources or create a documented source subset before labelling")
        source_ids = [s["source_id"] for s in sources]
        source_positions = {source_id: i for i, source_id in enumerate(source_ids)}
        graph, adjacency = [], []
        assertions = evidence.get("graph_assertions")
        if assertions is None:
            assertions = [dict(assertion, support_sets=[[source["source_id"]]])
                          for source in sources for assertion in source.get("assertions", [])]
        for assertion in assertions:
            # Flatten observable physical dependencies into incidence only. The
            # model does not receive the DNF groups or any claim-matched oracle.
            dependencies = {source_id for group in assertion.get("support_sets", []) for source_id in group}
            if not dependencies and assertion.get("source_id") is not None:
                dependencies = {assertion["source_id"]}
            if not dependencies or dependencies - source_positions.keys():
                raise ValueError("Graph assertion must depend on surviving, known sources")
            graph.append(assertion_text(assertion))
            adjacency.append([float(source_id in dependencies) for source_id in source_ids])
        # Patent [0087]/[0048]: rebuilt cache text forms ADDITIONAL graph nodes with a
        # "CACHE" prefix, alongside (not as) relation nodes.  It adds no independent
        # evidence; its incidence inherits the upstream source dependencies.
        for artifact in evidence.get("artifacts") or []:
            if artifact.get("channel") != "cache":
                continue
            payload = artifact.get("payload") if isinstance(artifact.get("payload"), dict) else {}
            text = payload.get("text") or ""
            if not text:
                continue
            dependencies = set(artifact.get("source_ids") or [])
            if not dependencies or dependencies - source_positions.keys():
                raise ValueError("Cache artifact must depend on surviving, known sources")
            graph.append("CACHE " + text)
            adjacency.append([float(source_id in dependencies) for source_id in source_ids])
        if len(graph) > self.config.get("max_graph_nodes", 256):
            raise ValueError("Too many graph nodes; raise max_graph_nodes or subset evidence before labelling")
        # Graph content includes derived facts but never oracle support states.
        claims = record["claims"]
        if not claims:
            raise ValueError("At least one claim query is required")
        claim_queries = [claim_text(c) for c in claims]
        return {
            "pair": self.pair_tokens(record, device),
            "source": self.tokens(["SOURCE " + s["source_id"] + " " + s["text"] for s in sources], self.config.get("source_max_length", 128), device) if sources else None,
            "graph": self.tokens(graph, self.config.get("graph_max_length", 96), device) if graph else None,
            "claim": self.tokens(claim_queries, self.config.get("claim_max_length", 96), device),
            "incidence": torch.tensor(adjacency, device=device).reshape(len(graph), len(sources)),
            "source_ids": source_ids,
            "claim_ids": [c["claim_id"] for c in claims],
        }
