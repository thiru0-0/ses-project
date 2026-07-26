"""
Azure DevOps MCP client — pulls work items, test cases, and wikis.

Connects to Azure DevOps Services (cloud) via its REST API.
Supports both the official MCP server flow and a direct REST fallback.

The module produces RawDocument instances that feed straight into
the existing normalizer → chunker → vector store pipeline.

Requires ADO_ORG_URL, ADO_PROJECT, and ADO_PAT in environment / .env.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from config.settings import settings
from src.ragpoc.ingestion.loaders import RawDocument

logger = logging.getLogger(__name__)

# ADO REST API version
API_VERSION = "7.1"


@dataclass
class ADOWorkItem:
    """Parsed representation of an Azure DevOps work item."""

    id: int
    title: str
    work_item_type: str  # "User Story", "Bug", "Task", "Test Case", etc.
    state: str
    description: str = ""
    acceptance_criteria: str = ""
    steps: str = ""  # for test cases: the test steps XML/HTML
    parent_id: int | None = None
    area_path: str = ""
    iteration_path: str = ""
    assigned_to: str = ""
    tags: str = ""
    url: str = ""
    metadata: dict = field(default_factory=dict)


class ADOMCPClient:
    """Client to pull data from Azure DevOps Services via REST API.

    This serves as the data bridge between ADO and the RAG pipeline.
    Each pull method returns a list of RawDocument objects that can
    be fed directly into the normalizer.
    """

    def __init__(
        self,
        org_url: str | None = None,
        project: str | None = None,
        pat: str | None = None,
    ):
        self._org_url = (org_url or settings.ado_org_url).rstrip("/")
        self._project = project or settings.ado_project
        self._pat = pat or settings.ado_pat

        if not self._org_url or not self._project or not self._pat:
            logger.warning(
                "ADO MCP client initialized with incomplete config. "
                "Set ADO_ORG_URL, ADO_PROJECT, and ADO_PAT."
            )

        # Build the session with PAT auth
        self._session = requests.Session()
        self._session.auth = ("", self._pat)
        self._session.headers.update(
            {"Content-Type": "application/json"}
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check connectivity to the ADO organization."""
        if not self._org_url or not self._pat:
            return False
        try:
            url = f"{self._org_url}/_apis/projects?api-version={API_VERSION}"
            resp = self._session.get(url, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.error("ADO health check failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Work item queries
    # ------------------------------------------------------------------

    def fetch_work_items_by_type(
        self, work_item_type: str, top: int = 200
    ) -> list[ADOWorkItem]:
        """Fetch work items of a given type via WIQL.

        Args:
            work_item_type: ADO type, e.g. "User Story", "Test Case", "Bug".
            top: Maximum number of items to return.

        Returns:
            List of parsed ADOWorkItem objects.
        """
        wiql = (
            f"SELECT [System.Id] FROM WorkItems "
            f"WHERE [System.TeamProject] = '{self._project}' "
            f"AND [System.WorkItemType] = '{work_item_type}' "
            f"ORDER BY [System.ChangedDate] DESC"
        )
        ids = self._run_wiql(wiql, top=top)
        if not ids:
            logger.info("No %s items found in project '%s'", work_item_type, self._project)
            return []

        return self._get_work_items_batch(ids)

    def fetch_all_work_items(self, top: int = 500) -> list[ADOWorkItem]:
        """Fetch all work items from the project.

        Args:
            top: Maximum number of items to return.

        Returns:
            List of parsed ADOWorkItem objects.
        """
        wiql = (
            f"SELECT [System.Id] FROM WorkItems "
            f"WHERE [System.TeamProject] = '{self._project}' "
            f"ORDER BY [System.ChangedDate] DESC"
        )
        ids = self._run_wiql(wiql, top=top)
        if not ids:
            return []

        return self._get_work_items_batch(ids)

    # ------------------------------------------------------------------
    # Wiki
    # ------------------------------------------------------------------

    def fetch_wiki_pages(self, wiki_identifier: str | None = None) -> list[RawDocument]:
        """Fetch all pages from an ADO wiki.

        Args:
            wiki_identifier: Name or ID of the wiki. If None, fetches
                             from the first project wiki found.

        Returns:
            List of RawDocument objects (source_type="ado_wiki").
        """
        # Discover wikis if no identifier given
        if not wiki_identifier:
            wikis = self._list_wikis()
            if not wikis:
                logger.info("No wikis found in project '%s'", self._project)
                return []
            wiki_identifier = wikis[0]["id"]

        # Get the wiki pages tree
        pages = self._get_wiki_pages(wiki_identifier)
        docs = []
        for page in pages:
            content = self._get_wiki_page_content(wiki_identifier, page["path"])
            if content and content.strip():
                docs.append(
                    RawDocument(
                        content=content,
                        source_type="ado_wiki",
                        source_ref=f"{self._org_url}/{self._project}/_wiki/wikis/{wiki_identifier}?pagePath={page['path']}",
                        metadata={
                            "title": page.get("path", "").split("/")[-1] or "Wiki Page",
                            "wiki_id": wiki_identifier,
                            "page_path": page.get("path", ""),
                            "ado_source": "wiki",
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )

        logger.info("Fetched %d wiki pages from wiki '%s'", len(docs), wiki_identifier)
        return docs

    # ------------------------------------------------------------------
    # Conversion to RawDocument
    # ------------------------------------------------------------------

    def work_items_to_raw_documents(
        self, items: list[ADOWorkItem]
    ) -> list[RawDocument]:
        """Convert ADOWorkItem objects into RawDocument for the ingestion pipeline.

        Each work item becomes a single RawDocument whose content is a
        structured text representation. ADO-specific fields are preserved
        in metadata.
        """
        docs = []
        for item in items:
            content = self._render_work_item_text(item)
            if not content.strip():
                continue

            docs.append(
                RawDocument(
                    content=content,
                    source_type="ado_work_item",
                    source_ref=item.url or f"{self._org_url}/{self._project}/_workitems/edit/{item.id}",
                    metadata={
                        "title": item.title,
                        "work_item_id": item.id,
                        "work_item_type": item.work_item_type,
                        "state": item.state,
                        "parent_id": item.parent_id or "",
                        "area_path": item.area_path,
                        "iteration_path": item.iteration_path,
                        "assigned_to": item.assigned_to,
                        "tags": item.tags,
                        "ado_source": "work_item",
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            )

        logger.info("Converted %d work items to RawDocuments", len(docs))
        return docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_wiql(self, wiql: str, top: int = 200) -> list[int]:
        """Execute a WIQL query and return matching work item IDs."""
        url = (
            f"{self._org_url}/{self._project}/_apis/wit/wiql"
            f"?api-version={API_VERSION}&$top={top}"
        )
        body = {"query": wiql}

        try:
            resp = self._session.post(url, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return [wi["id"] for wi in data.get("workItems", [])]
        except Exception as e:
            logger.error("WIQL query failed: %s", e)
            return []

    def _get_work_items_batch(self, ids: list[int]) -> list[ADOWorkItem]:
        """Fetch full work item details in batches of 200 (API limit)."""
        all_items = []
        batch_size = 200

        fields = [
            "System.Id",
            "System.Title",
            "System.WorkItemType",
            "System.State",
            "System.Description",
            "Microsoft.VSTS.Common.AcceptanceCriteria",
            "Microsoft.VSTS.TCM.Steps",
            "System.Parent",
            "System.AreaPath",
            "System.IterationPath",
            "System.AssignedTo",
            "System.Tags",
        ]

        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            url = (
                f"{self._org_url}/{self._project}/_apis/wit/workitems"
                f"?ids={','.join(str(x) for x in batch)}"
                f"&fields={','.join(fields)}"
                f"&api-version={API_VERSION}"
            )
            try:
                resp = self._session.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                for wi in data.get("value", []):
                    f = wi.get("fields", {})
                    assigned = f.get("System.AssignedTo", "")
                    if isinstance(assigned, dict):
                        assigned = assigned.get("displayName", "")

                    all_items.append(
                        ADOWorkItem(
                            id=f.get("System.Id", wi.get("id", 0)),
                            title=f.get("System.Title", ""),
                            work_item_type=f.get("System.WorkItemType", ""),
                            state=f.get("System.State", ""),
                            description=f.get("System.Description", ""),
                            acceptance_criteria=f.get(
                                "Microsoft.VSTS.Common.AcceptanceCriteria", ""
                            ),
                            steps=f.get("Microsoft.VSTS.TCM.Steps", ""),
                            parent_id=f.get("System.Parent"),
                            area_path=f.get("System.AreaPath", ""),
                            iteration_path=f.get("System.IterationPath", ""),
                            assigned_to=assigned,
                            tags=f.get("System.Tags", ""),
                            url=wi.get("url", ""),
                        )
                    )
            except Exception as e:
                logger.error("Batch fetch failed for IDs %s: %s", batch[:5], e)

        logger.info("Fetched %d work items in detail", len(all_items))
        return all_items

    def _list_wikis(self) -> list[dict]:
        """List all wikis in the project."""
        url = (
            f"{self._org_url}/{self._project}/_apis/wiki/wikis"
            f"?api-version={API_VERSION}"
        )
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json().get("value", [])
        except Exception as e:
            logger.error("Failed to list wikis: %s", e)
            return []

    def _get_wiki_pages(self, wiki_id: str) -> list[dict]:
        """Get the flat list of pages from a wiki (recursively)."""
        url = (
            f"{self._org_url}/{self._project}/_apis/wiki/wikis/{wiki_id}/pages"
            f"?recursionLevel=full&api-version={API_VERSION}"
        )
        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            root = resp.json()
            return self._flatten_wiki_tree(root)
        except Exception as e:
            logger.error("Failed to get wiki pages: %s", e)
            return []

    def _flatten_wiki_tree(self, node: dict) -> list[dict]:
        """Recursively flatten a wiki page tree into a list."""
        pages = []
        if node.get("path"):
            pages.append({"path": node["path"], "id": node.get("id", "")})
        for child in node.get("subPages", []):
            pages.extend(self._flatten_wiki_tree(child))
        return pages

    def _get_wiki_page_content(self, wiki_id: str, page_path: str) -> str:
        """Get the markdown content of a single wiki page."""
        import urllib.parse

        encoded_path = urllib.parse.quote(page_path, safe="")
        url = (
            f"{self._org_url}/{self._project}/_apis/wiki/wikis/{wiki_id}/pages"
            f"?path={encoded_path}&includeContent=true&api-version={API_VERSION}"
        )
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json().get("content", "")
        except Exception as e:
            logger.error("Failed to get wiki page '%s': %s", page_path, e)
            return ""

    @staticmethod
    def _render_work_item_text(item: ADOWorkItem) -> str:
        """Render a work item into structured plain text for embedding.

        Strips HTML tags from description / acceptance criteria / steps
        and formats the item as a clean, embeddable document.
        """
        import re

        def strip_html(html: str) -> str:
            if not html:
                return ""
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        parts = [
            f"[{item.work_item_type}] #{item.id}: {item.title}",
            f"State: {item.state}",
        ]

        if item.area_path:
            parts.append(f"Area: {item.area_path}")
        if item.iteration_path:
            parts.append(f"Iteration: {item.iteration_path}")
        if item.tags:
            parts.append(f"Tags: {item.tags}")
        if item.assigned_to:
            parts.append(f"Assigned to: {item.assigned_to}")
        if item.parent_id:
            parts.append(f"Parent: #{item.parent_id}")

        desc = strip_html(item.description)
        if desc:
            parts.append(f"\nDescription:\n{desc}")

        ac = strip_html(item.acceptance_criteria)
        if ac:
            parts.append(f"\nAcceptance Criteria:\n{ac}")

        steps = strip_html(item.steps)
        if steps:
            parts.append(f"\nTest Steps:\n{steps}")

        return "\n".join(parts)
