#!/usr/bin/env python3
"""
HQC AUDIT PATCH MIGRATION SCRIPT
=================================

Automated tool to apply all critical patches from the security audit.
Includes backup, validation, and rollback capability.

Usage:
    python apply_audit_patches.py              # Apply all patches
    python apply_audit_patches.py --validate   # Validate patches only
    python apply_audit_patches.py --rollback   # Rollback to backups
    python apply_audit_patches.py --list       # List all patches
    python apply_audit_patches.py --help       # Show help

Created: 2026-03-10
Author: HQC Audit Team
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AuditMigration")


class PatchManager:
    """Manages patch application, validation, and rollback."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.backup_dir = project_root / ".audit_backups"
        self.migration_log = project_root / ".audit_migration.json"
        self.patches: List[Patch] = []
        
        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(exist_ok=True)
        
        logger.info("Project root: %s", self.project_root)
        logger.info("Backup directory: %s", self.backup_dir)

    def register_patch(self, patch: Patch) -> None:
        """Register a patch to be applied."""
        self.patches.append(patch)

    def validate_environment(self) -> bool:
        """Validate that the project structure is correct."""
        logger.info("Validating project environment...")
        
        required_files = [
            "main.py",
            "core/engine/event_bus.py",
            "core/engine/state_machine.py",
            "intelligence/candidate_ranker.py",
            "strategies/vwap/hunter_state_machine.py",
            "parameter_optimizer.py",
            "data/feeds/ws_manager.py",
        ]
        
        missing = []
        for file in required_files:
            if not (self.project_root / file).exists():
                missing.append(file)
                logger.error("❌ Missing: %s", file)
            else:
                logger.debug("✓ Found: %s", file)
        
        if missing:
            logger.error("❌ Project validation FAILED. Missing %d files.", len(missing))
            return False
        
        logger.info("✅ Project validation PASSED.")
        return True

    def create_backup(self, file_path: Path) -> bool:
        """Create a timestamped backup of a file."""
        if not file_path.exists():
            logger.error("❌ Cannot backup non-existent file: %s", file_path)
            return False
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        relative_path = file_path.relative_to(self.project_root)
        backup_path = self.backup_dir / f"{relative_path.name}_{timestamp}.bak"
        
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_path)
            logger.info("✓ Backed up: %s → %s", relative_path, backup_path)
            return True
        except Exception as e:
            logger.error("❌ Backup failed for %s: %s", file_path, e)
            return False

    def apply_all_patches(self, dry_run: bool = False) -> Tuple[int, int, List[str]]:
        """
        Apply all registered patches.
        
        Returns:
            (success_count, failure_count, failed_patch_names)
        """
        success = 0
        failures = 0
        failed_names = []
        
        logger.info("=" * 70)
        logger.info("STARTING PATCH MIGRATION (%s)", "DRY RUN" if dry_run else "LIVE")
        logger.info("=" * 70)
        
        for i, patch in enumerate(self.patches, 1):
            logger.info("\n[%d/%d] Applying: %s", i, len(self.patches), patch.name)
            logger.info("Description: %s", patch.description)
            
            try:
                if not dry_run:
                    # Create backup before applying
                    if patch.file_path and patch.file_path.exists():
                        if not self.create_backup(patch.file_path):
                            logger.warning("⚠️  Backup creation failed, continuing anyway...")
                
                result = patch.apply(dry_run=dry_run)
                
                if result:
                    logger.info("✅ PASSED: %s", patch.name)
                    success += 1
                else:
                    logger.error("❌ FAILED: %s", patch.name)
                    failures += 1
                    failed_names.append(patch.name)
                    
            except Exception as e:
                logger.error("❌ EXCEPTION in %s: %s", patch.name, e, exc_info=True)
                failures += 1
                failed_names.append(patch.name)
        
        logger.info("\n" + "=" * 70)
        logger.info("MIGRATION COMPLETE: %d succeeded, %d failed", success, failures)
        logger.info("=" * 70)
        
        # Save migration log
        if not dry_run:
            self._save_migration_log(success, failures, failed_names)
        
        return success, failures, failed_names

    def _save_migration_log(self, success: int, failures: int, failed_names: List[str]) -> None:
        """Save migration results to JSON log."""
        log_data = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "success_count": success,
            "failure_count": failures,
            "failed_patches": failed_names,
            "total_patches": len(self.patches),
            "patches_applied": [p.name for p in self.patches[:success]],
        }
        
        try:
            self.migration_log.write_text(json.dumps(log_data, indent=2))
            logger.info("✓ Migration log saved: %s", self.migration_log)
        except Exception as e:
            logger.error("❌ Failed to save migration log: %s", e)

    def list_patches(self) -> None:
        """List all registered patches."""
        logger.info("=" * 70)
        logger.info("REGISTERED PATCHES (%d total)", len(self.patches))
        logger.info("=" * 70)
        
        for i, patch in enumerate(self.patches, 1):
            logger.info("\n[%d] %s", i, patch.name)
            logger.info("    File: %s", patch.file_path.relative_to(self.project_root) if patch.file_path else "N/A")
            logger.info("    Desc: %s", patch.description)
            logger.info("    Type: %s", patch.patch_type)

    def rollback_latest(self) -> bool:
        """Rollback to the most recent backup."""
        logger.info("=" * 70)
        logger.info("ROLLBACK INITIATED")
        logger.info("=" * 70)
        
        if not self.backup_dir.exists():
            logger.error("❌ No backup directory found.")
            return False
        
        backups = list(self.backup_dir.glob("*.bak"))
        if not backups:
            logger.error("❌ No backups found.")
            return False
        
        # Get most recent backup
        latest_backup = sorted(backups, key=lambda p: p.stat().st_mtime)[-1]
        
        # Extract original filename
        original_name = latest_backup.name.rsplit("_", 1)[0]
        
        # Find original file
        original_path = None
        for patch in self.patches:
            if patch.file_path and patch.file_path.name == original_name:
                original_path = patch.file_path
                break
        
        if not original_path:
            # Try to find it in project root
            potential = self.project_root / original_name
            if potential.exists():
                original_path = potential
        
        if not original_path:
            logger.error("❌ Could not locate original file for: %s", original_name)
            return False
        
        try:
            shutil.copy2(latest_backup, original_path)
            logger.info("✅ Restored: %s from %s", original_path.relative_to(self.project_root), latest_backup.name)
            return True
        except Exception as e:
            logger.error("❌ Rollback failed: %s", e)
            return False


class Patch:
    """Base class for a patch."""

    def __init__(
        self,
        name: str,
        description: str,
        file_path: Optional[Path] = None,
        patch_type: str = "modification",
    ):
        self.name = name
        self.description = description
        self.file_path = file_path
        self.patch_type = patch_type  # 'modification', 'addition', 'new_file'

    def apply(self, dry_run: bool = False) -> bool:
        """Apply the patch. Override in subclasses."""
        raise NotImplementedError


class TextReplacementPatch(Patch):
    """Patch that replaces text in a file."""

    def __init__(
        self,
        name: str,
        description: str,
        file_path: Path,
        search_text: str,
        replacement_text: str,
        patch_type: str = "modification",
    ):
        super().__init__(name, description, file_path, patch_type)
        self.search_text = search_text
        self.replacement_text = replacement_text

    def apply(self, dry_run: bool = False) -> bool:
        """Apply text replacement patch."""
        if not self.file_path.exists():
            logger.error("❌ File not found: %s", self.file_path)
            return False
        
        try:
            content = self.file_path.read_text(encoding="utf-8")
            
            if self.search_text not in content:
                logger.error(
                    "❌ Search text not found in %s",
                    self.file_path.relative_to(self.file_path.parent.parent.parent),
                )
                logger.debug("  Search text: %s...", self.search_text[:100])
                return False
            
            if not dry_run:
                new_content = content.replace(self.search_text, self.replacement_text)
                self.file_path.write_text(new_content, encoding="utf-8")
                logger.info("✓ Applied text replacement: %d chars changed", 
                           len(new_content) - len(content))
            else:
                logger.info("✓ [DRY RUN] Would apply text replacement")
            
            return True
            
        except Exception as e:
            logger.error("❌ Error applying patch: %s", e)
            return False


class InsertionPatch(Patch):
    """Patch that inserts code at a specific location."""

    def __init__(
        self,
        name: str,
        description: str,
        file_path: Path,
        anchor_text: str,
        insertion_text: str,
        position: str = "after",  # 'before' or 'after'
    ):
        super().__init__(name, description, file_path, "modification")
        self.anchor_text = anchor_text
        self.insertion_text = insertion_text
        self.position = position

    def apply(self, dry_run: bool = False) -> bool:
        """Apply insertion patch."""
        if not self.file_path.exists():
            logger.error("❌ File not found: %s", self.file_path)
            return False
        
        try:
            content = self.file_path.read_text(encoding="utf-8")
            
            if self.anchor_text not in content:
                logger.error("❌ Anchor text not found in %s", self.file_path.name)
                logger.debug("  Anchor text: %s...", self.anchor_text[:100])
                return False
            
            if not dry_run:
                if self.position == "after":
                    anchor_pos = content.find(self.anchor_text)
                    anchor_end = anchor_pos + len(self.anchor_text)
                    new_content = content[:anchor_end] + "\n" + self.insertion_text + content[anchor_end:]
                else:  # before
                    anchor_pos = content.find(self.anchor_text)
                    new_content = content[:anchor_pos] + self.insertion_text + "\n" + content[anchor_pos:]
                
                self.file_path.write_text(new_content, encoding="utf-8")
                logger.info("✓ Inserted %d chars at position '%s'", len(self.insertion_text), self.position)
            else:
                logger.info("✓ [DRY RUN] Would insert code at position '%s'", self.position)
            
            return True
            
        except Exception as e:
            logger.error("❌ Error applying insertion patch: %s", e)
            return False


class NewFilePatch(Patch):
    """Patch that creates a new file."""

    def __init__(
        self,
        name: str,
        description: str,
        file_path: Path,
        content: str,
    ):
        super().__init__(name, description, file_path, "new_file")
        self.content = content

    def apply(self, dry_run: bool = False) -> bool:
        """Create new file."""
        if self.file_path.exists():
            logger.warning("⚠️  File already exists: %s (will skip)", self.file_path.name)
            return True
        
        try:
            if not dry_run:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                self.file_path.write_text(self.content, encoding="utf-8")
                logger.info("✓ Created new file: %s (%d bytes)", self.file_path.name, len(self.content))
            else:
                logger.info("✓ [DRY RUN] Would create new file: %s (%d bytes)", 
                           self.file_path.name, len(self.content))
            
            return True
            
        except Exception as e:
            logger.error("❌ Error creating file: %s", e)
            return False


def build_patches(project_root: Path) -> List[Patch]:
    """Build all patches to be applied."""
    patches: List[Patch] = []
    
    # =========================================================================
    # PATCH 1: EventBus Timestamp Validation Helper
    # =========================================================================
    patches.append(
        InsertionPatch(
            name="P1-EventBus-Timestamp-Validation",
            description="Add timestamp normalization/validation helper to EventBus",
            file_path=project_root / "core/engine/event_bus.py",
            anchor_text="import asyncio\nimport logging\nfrom dataclasses import dataclass\nfrom enum import Enum, auto\nfrom typing import Any, Callable, Coroutine, Dict, List, Optional",
            insertion_text="import time",
            position="after",
        )
    )
    
    patches.append(
        InsertionPatch(
            name="P1b-EventBus-Validation-Method",
            description="Add validate_timestamp static method to Event class",
            file_path=project_root / "core/engine/event_bus.py",
            anchor_text="@dataclass\nclass Event:\n    type: EventType\n    payload: Dict[str, Any]",
            insertion_text='''
    @staticmethod
    def validate_timestamp(ts_value) -> int:
        """Normalizes and validates a timestamp to milliseconds."""
        if ts_value is None:
            return int(time.time() * 1000)
        
        # Already in milliseconds
        if isinstance(ts_value, int) and ts_value > 1_000_000_000_000:
            return ts_value
        
        # Seconds to milliseconds
        if isinstance(ts_value, (int, float)) and ts_value < 1_000_000_000_000:
            return int(ts_value * 1000)
        
        # ISO string format
        if isinstance(ts_value, str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            except Exception as e:
                raise ValueError(f"Cannot parse timestamp string: {ts_value}") from e
        
        raise ValueError(f"Invalid timestamp type: {type(ts_value)}")''',
            position="after",
        )
    )
    
    # =========================================================================
    # PATCH 2: Main.py - Feed Stale-Data Detection
    # =========================================================================
    patches.append(
        InsertionPatch(
            name="P2-Main-Imports",
            description="Add time import to main.py",
            file_path=project_root / "main.py",
            anchor_text="from dotenv import load_dotenv",
            insertion_text="import time",
            position="after",
        )
    )
    
    patches.append(
        InsertionPatch(
            name="P2b-Main-StaleData-Init",
            description="Add stale-data detection fields to TradingNode.__init__",
            file_path=project_root / "main.py",
            anchor_text="        self.bus = EventBus()\n\n        universe_env = os.getenv(\"HQC_UNIVERSE\", \"SPY,QQQ,TSLA\")",
            insertion_text="""        # --- PATCH: Stale feed detection ---
        self._last_tick_timestamp_ms: int = int(asyncio.get_event_loop().time() * 1000)
        self._max_tick_staleness_sec: float = 60.0  # Halt if no tick for 60 seconds
        self._stale_check_task: Optional[asyncio.Task] = None
        # --- END PATCH ---""",
            position="before",
        )
    )
    
    patches.append(
        InsertionPatch(
            name="P2c-Main-StaleDataMethod",
            description="Add _check_feed_staleness method to TradingNode",
            file_path=project_root / "main.py",
            anchor_text="    async def on_tick(self, event: Event) -> None:",
            insertion_text="""    async def _check_feed_staleness(self) -> None:
        \"\"\"
        Background task to detect stale market data.
        If no tick received for max_tick_staleness_sec, halt the system.
        \"\"\"
        while self.state_machine.current_state != SystemState.HALTED and not self._shutdown_started:
            await asyncio.sleep(5.0)  # Check every 5 seconds
            
            current_time_ms = int(time.time() * 1000)
            staleness_ms = current_time_ms - self._last_tick_timestamp_ms
            staleness_sec = staleness_ms / 1000.0
            
            if staleness_sec > self._max_tick_staleness_sec:
                logger.error(
                    "Data feed is STALE. No tick received for %.1f seconds. "
                    "Last tick symbol: %s at %s",
                    staleness_sec,
                    self._last_tick_symbol,
                    self._last_tick_ts,
                )
                self.state_machine.transition_to(
                    SystemState.HALTED,
                    f"Data feed stale for {staleness_sec:.1f}s (threshold: {self._max_tick_staleness_sec}s)",
                )
                break

    """,
            position="before",
        )
    )
    
    patches.append(
        TextReplacementPatch(
            name="P2d-Main-OnTick",
            description="Update on_tick to record timestamp for stale detection",
            file_path=project_root / "main.py",
            search_text="""    async def on_tick(self, event: Event) -> None:
        payload = event.payload or {}
        self._last_tick_symbol = payload.get("ticker") or payload.get("symbol")
        ts_val = payload.get("timestamp")
        self._last_tick_ts = str(ts_val) if ts_val is not None else None""",
            replacement_text="""    async def on_tick(self, event: Event) -> None:
        payload = event.payload or {}
        self._last_tick_symbol = payload.get("ticker") or payload.get("symbol")
        ts_val = payload.get("timestamp")
        self._last_tick_ts = str(ts_val) if ts_val is not None else None
        
        # --- PATCH: Update stale-data detection timestamp ---
        try:
            ts_ms = event.validate_timestamp(ts_val) if hasattr(event, 'validate_timestamp') else int(time.time() * 1000)
            self._last_tick_timestamp_ms = ts_ms
        except Exception as e:
            logger.warning("Failed to parse tick timestamp: %s. Using system time.", e)
            self._last_tick_timestamp_ms = int(time.time() * 1000)
        # --- END PATCH ---""",
        )
    )
    
    patches.append(
        TextReplacementPatch(
            name="P2e-Main-StartMethod",
            description="Add stale-check task to start() method",
            file_path=project_root / "main.py",
            search_text="""        self._feed_task = asyncio.create_task(self.data_feed.start(), name="tradier_data_feed")
        self._health_task = asyncio.create_task(self._health_loop(), name="feedback_health_loop")""",
            replacement_text="""        self._feed_task = asyncio.create_task(self.data_feed.start(), name="tradier_data_feed")
        # --- PATCH: Add staleness detection task ---
        self._stale_check_task = asyncio.create_task(self._check_feed_staleness(), name="feed_staleness_monitor")
        # --- END PATCH ---
        self._health_task = asyncio.create_task(self._health_loop(), name="feedback_health_loop")""",
        )
    )
    
    patches.append(
        TextReplacementPatch(
            name="P2f-Main-ShutdownMethod",
            description="Add stale-check task cleanup to shutdown() method",
            file_path=project_root / "main.py",
            search_text="""        if self._feed_task is not None:
            self._feed_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._feed_task

        if self._health_task is not None:""",
            replacement_text="""        if self._feed_task is not None:
            self._feed_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._feed_task

        # --- PATCH: Cancel staleness check task ---
        if hasattr(self, '_stale_check_task') and self._stale_check_task is not None:
            self._stale_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._stale_check_task
        # --- END PATCH ---

        if self._health_task is not None:""",
        )
    )
    
    # =========================================================================
    # PATCH 3: Main.py - WARMING_UP Timeout
    # =========================================================================
    patches.append(
        InsertionPatch(
            name="P3-Main-WarmingUpInit",
            description="Add WARMING_UP timeout fields to TradingNode.__init__",
            file_path=project_root / "main.py",
            anchor_text="        self._stale_check_task: Optional[asyncio.Task] = None\n        # --- END PATCH ---",
            insertion_text="""
        # --- PATCH: WARMING_UP timeout protection ---
        self._warming_up_start_time: Optional[float] = None
        self._warming_up_timeout_sec: float = 300.0  # 5 minutes
        self._warming_up_check_task: Optional[asyncio.Task] = None
        # --- END PATCH ---""",
            position="after",
        )
    )
    
    patches.append(
        InsertionPatch(
            name="P3b-Main-WarmingUpMethod",
            description="Add _check_warming_up_timeout method to TradingNode",
            file_path=project_root / "main.py",
            anchor_text="    async def _check_feed_staleness(self) -> None:",
            insertion_text="""    async def _check_warming_up_timeout(self) -> None:
        \"\"\"
        If system stays in WARMING_UP for too long without receiving first tick,
        automatically halt to prevent hung startup.
        \"\"\"
        while self.state_machine.current_state != SystemState.HALTED and not self._shutdown_started:
            await asyncio.sleep(10.0)  # Check every 10 seconds
            
            if self.state_machine.current_state == SystemState.WARMING_UP:
                if self._warming_up_start_time is None:
                    self._warming_up_start_time = asyncio.get_event_loop().time()
                
                elapsed_sec = asyncio.get_event_loop().time() - self._warming_up_start_time
                
                if elapsed_sec > self._warming_up_timeout_sec:
                    logger.error(
                        "System stuck in WARMING_UP for %.1f seconds. "
                        "No market data received. Auto-halting.",
                        elapsed_sec,
                    )
                    self.state_machine.transition_to(
                        SystemState.HALTED,
                        f"WARMING_UP timeout after {elapsed_sec:.1f}s (no first tick)",
                    )
                    break
            else:
                # System progressed out of WARMING_UP, reset timer
                self._warming_up_start_time = None

    """,
            position="before",
        )
    )
    
    patches.append(
        TextReplacementPatch(
            name="P3c-Main-StartWarmingUpTimer",
            description="Initialize warming-up timer in start() method",
            file_path=project_root / "main.py",
            search_text="""        self.state_machine.transition_to(SystemState.WARMING_UP, "Services initialized.")

        self._feed_task = asyncio.create_task(self.data_feed.start(), name="tradier_data_feed")""",
            replacement_text="""        self.state_machine.transition_to(SystemState.WARMING_UP, "Services initialized.")
        self._warming_up_start_time = asyncio.get_event_loop().time()  # --- PATCH ---

        self._feed_task = asyncio.create_task(self.data_feed.start(), name="tradier_data_feed")""",
        )
    )
    
    patches.append(
        TextReplacementPatch(
            name="P3d-Main-StartWarmingUpTask",
            description="Add warming-up timeout check task to start() method",
            file_path=project_root / "main.py",
            search_text="""        self._stale_check_task = asyncio.create_task(self._check_feed_staleness(), name="feed_staleness_monitor")
        # --- END PATCH ---
        self._health_task = asyncio.create_task(self._health_loop(), name="feedback_health_loop")""",
            replacement_text="""        self._stale_check_task = asyncio.create_task(self._check_feed_staleness(), name="feed_staleness_monitor")
        # --- END PATCH ---
        # --- PATCH: Add warming-up timeout check ---
        self._warming_up_check_task = asyncio.create_task(self._check_warming_up_timeout(), name="warming_up_timeout_monitor")
        # --- END PATCH ---
        self._health_task = asyncio.create_task(self._health_loop(), name="feedback_health_loop")""",
        )
    )
    
    patches.append(
        TextReplacementPatch(
            name="P3e-Main-ShutdownWarmingUp",
            description="Add warming-up task cleanup to shutdown() method",
            file_path=project_root / "main.py",
            search_text="""        # --- END PATCH ---

        if self._health_task is not None:
            self._health_task.cancel()""",
            replacement_text="""        # --- END PATCH ---

        # --- PATCH: Cancel warming-up timeout check ---
        if hasattr(self, '_warming_up_check_task') and self._warming_up_check_task is not None:
            self._warming_up_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._warming_up_check_task
        # --- END PATCH ---

        if self._health_task is not None:
            self._health_task.cancel()""",
        )
    )
    
    # =========================================================================
    # PATCH 4: Main.py - TRADING_MODE Validation
    # =========================================================================
    patches.append(
        TextReplacementPatch(
            name="P4-Main-TradingModeValidation",
            description="Add TRADING_MODE environment validation",
            file_path=project_root / "main.py",
            search_text="""        self.initial_capital = float(os.getenv("HQC_INITIAL_CAPITAL", "100000"))
        
        # Execution Keys
        self.api_key = os.getenv("ALPACA_API_KEY", os.getenv("APCA_API_KEY_ID", "YOUR_PAPER_KEY"))
        self.api_secret = os.getenv("ALPACA_API_SECRET", os.getenv("APCA_API_SECRET_KEY", "YOUR_PAPER_SECRET"))
        self.is_paper = os.getenv("TRADING_MODE", "PAPER").upper() == "PAPER\"""",
            replacement_text="""        self.initial_capital = float(os.getenv("HQC_INITIAL_CAPITAL", "100000"))
        
        # --- PATCH: Validate TRADING_MODE ---
        trading_mode_raw = os.getenv("TRADING_MODE", "PAPER").upper().strip()
        valid_modes = {"PAPER", "LIVE"}
        if trading_mode_raw not in valid_modes:
            raise ValueError(
                f"Invalid TRADING_MODE={trading_mode_raw}. "
                f"Must be one of {valid_modes}. Check your .env file."
            )
        self.is_paper = trading_mode_raw == "PAPER"
        logger.info("TRADING_MODE validated: %s", trading_mode_raw)
        # --- END PATCH ---
        
        # Execution Keys
        self.api_key = os.getenv("ALPACA_API_KEY", os.getenv("APCA_API_KEY_ID", "YOUR_PAPER_KEY"))
        self.api_secret = os.getenv("ALPACA_API_SECRET", os.getenv("APCA_API_SECRET_KEY", "YOUR_PAPER_SECRET\"))""",
        )
    )
    
    # =========================================================================
    # PATCH 5: CandidateRanker - VWAP Scorer Edge Case
    # =========================================================================
    patches.append(
        TextReplacementPatch(
            name="P5-CandidateRanker-VWAPEdgeCase",
            description="Fix VWAP scorer edge case at exactly 0%",
            file_path=project_root / "intelligence/candidate_ranker.py",
            search_text="""        # Asymmetric VWAP Logic
        # If buying, we want it above VWAP (positive dist). If negative, it's lagging.
        # If shorting, we want it below VWAP (negative dist). If positive, it's stubbornly strong.
        is_aligned_with_vwap = (action == "BUY" and dist_vwap_pct > 0) or (action in {"SELL", "SELL_SHORT"} and dist_vwap_pct < 0)""",
            replacement_text="""        # Asymmetric VWAP Logic
        # If buying, we want it above VWAP (positive dist). If negative, it's lagging.
        # If shorting, we want it below VWAP (negative dist). If positive, it's stubbornly strong.
        # --- PATCH: Fixed edge case at exactly 0% ---
        is_aligned_with_vwap = (action == "BUY" and dist_vwap_pct >= 0) or (action in {"SELL", "SELL_SHORT"} and dist_vwap_pct <= 0)
        # --- END PATCH: Changed > to >= and < to <= for boundary tolerance ---""",
        )
    )
    
    # =========================================================================
    # PATCH 6: HunterStateMachine - Window Staleness Fix
    # =========================================================================
    patches.append(
        TextReplacementPatch(
            name="P6a-Hunter-MaxWindowBarsParam",
            description="Add max_window_bars parameter to USEquityVWAPHunter.__init__",
            file_path=project_root / "strategies/vwap/hunter_state_machine.py",
            search_text="""        cooldown_bars: int = 8,
        min_stop_pct: float = 0.003,
    ) -> None:""",
            replacement_text="""        cooldown_bars: int = 8,
        min_stop_pct: float = 0.003,
        # --- PATCH: Add configurable window timeout ---
        max_window_bars: int = 8,
        # --- END PATCH ---
    ) -> None:""",
        )
    )
    
    patches.append(
        TextReplacementPatch(
            name="P6b-Hunter-AssignMaxWindowBars",
            description="Assign max_window_bars in USEquityVWAPHunter.__init__",
            file_path=project_root / "strategies/vwap/hunter_state_machine.py",
            search_text="""        self.max_daily_trades = int(max_daily_trades)
        self.cooldown_bars = int(cooldown_bars)
        self.min_stop_pct = float(min_stop_pct)

        self.tz = pytz.timezone("US/Eastern")""",
            replacement_text="""        self.max_daily_trades = int(max_daily_trades)
        self.cooldown_bars = int(cooldown_bars)
        self.min_stop_pct = float(min_stop_pct)
        self.max_window_bars = int(max_window_bars)  # --- PATCH ---

        self.tz = pytz.timezone("US/Eastern")""",
        )
    )
    
    patches.append(
        TextReplacementPatch(
            name="P6c-Hunter-WindowTimeout",
            description="Use configurable max_window_bars in window staleness check",
            file_path=project_root / "strategies/vwap/hunter_state_machine.py",
            search_text="""            # stale window reset
            if (self.bar_count - self.window_open_bar) > 20:
                logger.info("[%s] VWAP window stale. Resetting to SCANNING.", self.asset)
                self.current_state = HunterState.SCANNING
                return""",
            replacement_text="""            # --- PATCH: Fixed to use configurable max_window_bars instead of hardcoded 20 ---
            if (self.bar_count - self.window_open_bar) > self.max_window_bars:
                logger.info(
                    "[%s] VWAP window stale after %d bars. Resetting to SCANNING.",
                    self.asset,
                    self.bar_count - self.window_open_bar,
                )
                self.current_state = HunterState.SCANNING
                return
            # --- END PATCH ---""",
        )
    )
    
    # =========================================================================
    # PATCH 7: ParameterOptimizer - Sizer Fallback Visibility
    # =========================================================================
    patches.append(
        TextReplacementPatch(
            name="P7-Optimizer-FallbackVisibility",
            description="Add explicit logging for sizer fallback in optimizer",
            file_path=project_root / "parameter_optimizer.py",
            search_text="""        logger.info(
            "[DEBUG] date=%s raw_orders=%d sized_orders=%d",
            date_str,
            len(raw_orders),
            len(sized_orders),
        )

        # Prefer sized orders, but fall back to raw orders if sizing path didn't surface
        orders_for_scoring = sized_orders if sized_orders else self._promote_raw_orders(""",
            replacement_text="""        logger.info(
            "[DEBUG] date=%s raw_orders=%d sized_orders=%d",
            date_str,
            len(raw_orders),
            len(sized_orders),
        )

        # --- PATCH: Explicit logging for sizer fallback ---
        if sized_orders:
            logger.info("[SCORING] Using %d SIZED orders from DynamicRiskSizer.", len(sized_orders))
            orders_for_scoring = sized_orders
        elif raw_orders:
            logger.warning(
                "[FALLBACK] DynamicRiskSizer produced 0 sized orders. "
                "Falling back to manual promotion of %d raw orders. "
                "This may mask issues in the sizer. Review capital/risk parameters.",
                len(raw_orders),
            )
            orders_for_scoring = self._promote_raw_orders(""",
        )
    )
    
    patches.append(
        TextReplacementPatch(
            name="P7b-Optimizer-FallbackCompletion",
            description="Complete sizer fallback logging logic",
            file_path=project_root / "parameter_optimizer.py",
            search_text="""            orders_for_scoring = self._promote_raw_orders(
            raw_orders=raw_orders,
            account_equity=100000.0,
            base_risk_pct=base_risk_pct,
            max_position_pct=max_position_pct,
        )""",
            replacement_text="""            orders_for_scoring = self._promote_raw_orders(
                raw_orders=raw_orders,
                account_equity=100000.0,
                base_risk_pct=base_risk_pct,
                max_position_pct=max_position_pct,
            )
        else:
            logger.warning("[NO ORDERS] Date %s produced 0 signals (raw or sized).", date_str)
            orders_for_scoring = []
        # --- END PATCH ---""",
        )
    )
    
    # =========================================================================
    # PATCH 8: WebSocketManager - Timestamp Normalization
    # =========================================================================
    patches.append(
        InsertionPatch(
            name="P8a-WebSocket-TimeImport",
            description="Add time import to ws_manager.py",
            file_path=project_root / "data/feeds/ws_manager.py",
            anchor_text="import asyncio\nimport json\nimport logging\nimport websockets",
            insertion_text="import time",
            position="after",
        )
    )
    
    patches.append(
        TextReplacementPatch(
            name="P8b-WebSocket-TimestampNormalization",
            description="Normalize WebSocket timestamp to milliseconds",
            file_path=project_root / "data/feeds/ws_manager.py",
            search_text="""                for event in data:
                    # 't' denotes a Trade event in Alpaca's IEX stream
                    if event.get("T") == "t":
                        tick_event = Event(
                            type=EventType.TICK,
                            payload={
                                "ticker": event.get("S"),       # Symbol (e.g., 'AAPL')
                                "price": float(event.get("p")), # Trade Price
                                "volume": float(event.get("s")),# Trade Size (shares)
                                "timestamp": event.get("t"),    # SIP Timestamp""",
            replacement_text="""                for event in data:
                    # 't' denotes a Trade event in Alpaca's IEX stream
                    if event.get("T") == "t":
                        # --- PATCH: Normalize timestamp to milliseconds ---
                        ts_raw = event.get("t")  # Alpaca provides in nanoseconds
                        if ts_raw is not None:
                            # Alpaca SIP timestamp is in nanoseconds, convert to ms
                            ts_ms = int(ts_raw / 1_000_000)
                        else:
                            ts_ms = int(time.time() * 1000)
                        # --- END PATCH ---
                        
                        tick_event = Event(
                            type=EventType.TICK,
                            payload={
                                "ticker": event.get("S"),       # Symbol (e.g., 'AAPL')
                                "price": float(event.get("p")), # Trade Price
                                "volume": float(event.get("s")),# Trade Size (shares)
                                "timestamp": ts_ms,             # --- PATCH: Now in ms ---""",
        )
    )
    
    # =========================================================================
    # PATCH 9: Create .env.example
    # =========================================================================
    env_example_content = """# ==============================================================================
# HQC (Hybrid Quantitative System) Environment Configuration
# ==============================================================================
# Copy this file to .env and fill in your actual credentials
# DO NOT commit .env to version control

# --- TRADING EXECUTION (Alpaca Broker) ---
# Paper trading: obtain from https://app.alpaca.markets/paper (free)
ALPACA_API_KEY=PK_YOUR_PAPER_KEY_HERE
ALPACA_API_SECRET=YOUR_PAPER_SECRET_HERE

# Trading mode: must be PAPER or LIVE (default: PAPER)
# PAPER = simulated trading (default, safe for testing)
# LIVE = real money execution (use with extreme caution)
TRADING_MODE=PAPER

# --- MARKET DATA (Tradier Broker SIP) ---
# Obtain from https://tradier.com/api (free tier available)
TRADIER_API_TOKEN=YOUR_TRADIER_TOKEN_HERE

# --- SYSTEM CONFIGURATION ---
# Comma-separated list of stock symbols to trade (default: SPY,QQQ,TSLA)
HQC_UNIVERSE=SPY,QQQ,TSLA

# Initial account capital in dollars (default: 100000)
HQC_INITIAL_CAPITAL=100000

# Minimum ranker score to approve trades (0-10 scale, default: 4.75)
HQC_MIN_SCORE=4.75

# Maximum bid-ask spread in basis points (default: 18)
HQC_MAX_SPREAD_BPS=18

# Maximum distance from VWAP as percentage (default: 0.012 = 1.2%)
HQC_MAX_DIST_VWAP=0.012

# Base risk per trade as % of account (default: 0.01 = 1%)
HQC_RISK_PCT=0.01

# Maximum position size as % of account (default: 0.20 = 20%)
HQC_MAX_POSITION_PCT=0.20

# Maximum acceptable slippage in basis points (default: 8)
HQC_MAX_SLIPPAGE_BPS=8

# Maximum time for order to fill before cancellation in seconds (default: 10)
HQC_MAX_HANGING_SEC=10

# --- PARAMETER OPTIMIZER (Optional) ---
# Asset to optimize strategies for (default: SPY)
HQC_OPT_ASSET=SPY

# How many years of historical data to use (default: 5)
HQC_OPT_YEARS_BACK=5

# How many random trading days per config to test (default: 5)
HQC_OPT_RUNS_PER_CONFIG=5

# Random seed for reproducibility (default: 42)
HQC_OPT_SEED=42

# ==============================================================================
# SECURITY NOTES
# ==============================================================================
# - Never commit this file with real credentials to GitHub
# - Use strong, unique API keys (not your trading password)
# - For LIVE trading, use separate API keys with limited permissions
# - Rotate credentials regularly
# - Monitor account for unexpected trading activity
# ==============================================================================
"""
    
    patches.append(
        NewFilePatch(
            name="P9-Create-EnvExample",
            description="Create .env.example documentation",
            file_path=project_root / ".env.example",
            content=env_example_content,
        )
    )
    
    # =========================================================================
    # PATCH 10: Add New Test Cases
    # =========================================================================
    test_code_addition = """
    async def test_bearish_orb_breakdown(self):
        \"\"\"Test short breakout signal generation below range low.\"\"\"
        base_date = datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)
        
        # Build range 501-502
        ticks_building = [
            self._generate_mock_tick(501.50, base_date),
            self._generate_mock_tick(502.00, base_date + timedelta(minutes=5)),
            self._generate_mock_tick(501.00, base_date + timedelta(minutes=10)),
            self._generate_mock_tick(501.50, base_date + timedelta(minutes=14)),
        ]
        
        for tick in ticks_building:
            await self._pump_and_wait(tick)
        
        self.assertEqual(self.orb.state, ORBState.BUILDING_RANGE)
        self.assertEqual(self.orb.range_high, 502.00)
        self.assertEqual(self.orb.range_low, 501.00)
        self.assertEqual(len(self.captured_orders), 0)

        # Range established
        await self._pump_and_wait(self._generate_mock_tick(501.50, base_date + timedelta(minutes=15)))
        self.assertEqual(self.orb.state, ORBState.ACTIVE)

        # BEARISH BREAKDOWN: Below range_low * (1 - buffer)
        # Trigger ~500.75. Push to 500.50 to safely clear.
        await self._pump_and_wait(self._generate_mock_tick(500.50, base_date + timedelta(minutes=18)))
        
        self.assertEqual(len(self.captured_orders), 1, "Strategy failed to fire on breakdown!")
        
        fired_order = self.captured_orders[0].payload
        self.assertEqual(fired_order["action"], "SELL_SHORT")
        self.assertEqual(fired_order["reference_price"], 500.50)
        self.assertGreater(fired_order["stop_loss_price"], fired_order["reference_price"])

    async def test_max_trades_limit_enforced(self):
        \"\"\"Verify ORB respects max_trades limit and stops firing after N trades.\"\"\"
        base_date = datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)
        
        # Re-create ORB with max_trades=1
        self.orb = USEquityORB(
            target_asset=self.asset,
            bus=self.bus,
            range_minutes=15,
            max_trades=1,  # Only 1 trade allowed
            breakout_buffer_pct=0.0005
        )
        self.captured_orders = []
        self.bus.subscribe(EventType.ORDER_CREATE, self._order_catcher)

        # Build tight range and move to ACTIVE
        ticks = [
            self._generate_mock_tick(500.00, base_date + timedelta(minutes=1)),
            self._generate_mock_tick(500.50, base_date + timedelta(minutes=5)),
            self._generate_mock_tick(500.25, base_date + timedelta(minutes=10)),
            self._generate_mock_tick(500.25, base_date + timedelta(minutes=15)),
        ]
        
        for tick in ticks:
            await self._pump_and_wait(tick)
        
        self.assertEqual(self.orb.state, ORBState.ACTIVE)
        self.assertEqual(self.orb.trades_today, 0)

        # First breakout (should fire)
        await self._pump_and_wait(self._generate_mock_tick(500.80, base_date + timedelta(minutes=18)))
        self.assertEqual(len(self.captured_orders), 1)
        self.assertEqual(self.orb.trades_today, 1)
        self.assertEqual(self.orb.state, ORBState.DONE_FOR_DAY)

        # Try second breakout (should NOT fire - already at max)
        await self._pump_and_wait(self._generate_mock_tick(500.85, base_date + timedelta(minutes=20)))
        self.assertEqual(len(self.captured_orders), 1, "Strategy fired 2nd trade despite max_trades=1")

    async def test_range_too_narrow_rejection(self):
        \"\"\"Verify ORB rejects ranges that are too narrow.\"\"\"
        base_date = datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)
        
        # Build extremely narrow range (1 bps) - below min_range_pct (25 bps)
        ticks_building = [
            self._generate_mock_tick(500.00, base_date),
            self._generate_mock_tick(500.005, base_date + timedelta(minutes=5)),
            self._generate_mock_tick(500.00, base_date + timedelta(minutes=10)),
            self._generate_mock_tick(500.00, base_date + timedelta(minutes=14)),
        ]
        
        for tick in ticks_building:
            await self._pump_and_wait(tick)
        
        # Move past range end time
        await self._pump_and_wait(self._generate_mock_tick(500.01, base_date + timedelta(minutes=16)))
        
        # Should reject as DONE_FOR_DAY, not move to ACTIVE
        self.assertEqual(self.orb.state, ORBState.DONE_FOR_DAY, "Should reject narrow range")
        self.assertEqual(len(self.captured_orders), 0, "Should not fire on invalid range")
"""
    
    patches.append(
        InsertionPatch(
            name="P10-TestBearishBreakdown",
            description="Add bearish breakdown test case",
            file_path=project_root / "tests/test_backtest_parity.py",
            anchor_text="""    async def test_pre_market_filtering(self):
        pre_market_date = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        await self._pump_and_wait(self._generate_mock_tick(600.00, pre_market_date))
        
        self.assertEqual(self.orb.state, ORBState.PRE_MARKET)
        self.assertEqual(self.orb.range_high, float('-inf'))""",
            insertion_text=test_code_addition,
            position="before",
        )
    )
    
    return patches


def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description="HQC Audit Patch Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python apply_audit_patches.py              # Apply all patches (live)
  python apply_audit_patches.py --validate   # Validate patches only
  python apply_audit_patches.py --rollback   # Rollback latest backup
  python apply_audit_patches.py --list       # List all patches
  python apply_audit_patches.py --dry-run    # Apply patches (no writes)
        """,
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate project environment and patches only (no changes)",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to most recent backup",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered patches",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apply patches in dry-run mode (no file writes)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to HQC project root (default: current directory)",
    )
    
    args = parser.parse_args()
    
    # Determine project root
    project_root = args.project_root.resolve()
    
    # Verify we're in the right place
    if not (project_root / "main.py").exists():
        logger.error(
            "❌ main.py not found in %s. Are you in the HQC project root?",
            project_root,
        )
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("HQC AUDIT PATCH MIGRATION TOOL")
    logger.info("=" * 70)
    logger.info("Project Root: %s", project_root)
    
    # Initialize patch manager
    manager = PatchManager(project_root)
    
    # Validate environment first
    if not manager.validate_environment():
        logger.error("❌ Environment validation failed. Aborting.")
        sys.exit(1)
    
    # Build all patches
    patches = build_patches(project_root)
    for patch in patches:
        manager.register_patch(patch)
    
    logger.info("✅ Built %d patches.", len(patches))
    
    # Handle different modes
    if args.list:
        manager.list_patches()
        sys.exit(0)
    
    if args.rollback:
        logger.info("=" * 70)
        logger.info("ROLLBACK MODE")
        logger.info("=" * 70)
        if manager.rollback_latest():
            logger.info("✅ Rollback completed successfully.")
            sys.exit(0)
        else:
            logger.error("❌ Rollback failed.")
            sys.exit(1)
    
    if args.validate:
        logger.info("=" * 70)
        logger.info("VALIDATION MODE")
        logger.info("=" * 70)
        logger.info("✅ All validations passed. Ready to apply patches.")
        sys.exit(0)
    
    # Apply patches
    dry_run = args.dry_run
    success, failures, failed_names = manager.apply_all_patches(dry_run=dry_run)
    
    if dry_run:
        logger.info("\n✅ DRY RUN COMPLETE. No files were modified.")
        logger.info("To apply patches for real, run: python apply_audit_patches.py")
        sys.exit(0)
    
    # Exit with appropriate code
    if failures > 0:
        logger.error("\n❌ Migration completed with %d failures.", failures)
        logger.error("Failed patches: %s", ", ".join(failed_names))
        sys.exit(1)
    else:
        logger.info("\n✅ All %d patches applied successfully!", success)
        logger.info("Next steps:")
        logger.info("  1. Review changes: git diff")
        logger.info("  2. Run tests: python -m unittest test_backtest_parity -v")
        logger.info("  3. Commit: git add . && git commit -m 'audit: Apply critical fixes'")
        sys.exit(0)


if __name__ == "__main__":
    main()
